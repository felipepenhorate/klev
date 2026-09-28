#!/usr/bin/env python
"""TypeSafe-compatible decision endpoint for a klev checkpoint (SPEC.md, `serve/server.py`).

    POST /v1/systemone   {"state": ..., "questions": {qid: {type, instructions, criteria}}}
                        -> {"answers": {qid: {type, ..., probabilities, confidence, none}}}
    GET  /v1/models      -> the loaded base + adapter + temperature
    GET  /health         -> {"ok": true}

Same routes and response shapes as kev's own server, so eval/eval_systemone_http.py can score it
on the bench records unmodified.

    python serve/server.py --ckpt lumierenoir/klev-0.8b --port 8090
    curl -s localhost:8090/v1/systemone -H 'content-type: application/json' -d @request.json

This is optional. A decision model is small and stateless, so an in-process call is usually
better -- see examples/system_one.py, which uses the identical model.infer.answer(). Reach for a
server when several processes share one warm copy of the weights, or when the caller is not
Python. It is stdlib http.server, deliberately: no fastapi/uvicorn dependency to install on a
box that is already awkward about kernels.

One decision per request, encoded and forwarded sequentially under a lock. Batching would help
throughput; it would not change a single returned probability.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATE = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, "model": STATE.get("name")})
        elif self.path.rstrip("/") == "/v1/models":
            self._send(200, {"data": [{"id": STATE.get("name"), "object": "model",
                                       "base": STATE.get("base"),
                                       "temperature": STATE.get("temperature"),
                                       "garbage_channel": STATE.get("garbage")}]})
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/systemone":
            return self._send(404, {"error": f"no route {self.path}"})
        length = int(self.headers.get("content-length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            if "state" not in request or "questions" not in request:
                raise ValueError("body needs 'state' and 'questions'")
            # One lock: the model is shared and a forward is not re-entrant.
            with STATE["lock"]:
                out = STATE["answer"](request)
        except Exception as e:  # a bad request is the caller's problem, not a 500 stack trace
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        self._send(200, out)

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} {fmt % args}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lumierenoir/klev-0.8b", help="Hub repo id or local dir")
    ap.add_argument("--base", default=None)
    ap.add_argument("--backend", default=None, choices=[None, "unsloth", "plain"],
                    help="base loader: unsloth FastModel (default) or plain transformers+bitsandbytes")
    ap.add_argument("--preset", default="qwen35-08b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 raw; the card's served numbers use 2.2974 (IT) / 2.2449 (-Base)")
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--name", default="")
    return ap.parse_args()


def main():
    a = parse_args()
    from model.infer import answer, load_klev

    print(f"[serve] loading {a.ckpt} (preset {a.preset}) ...", flush=True)
    model, tokenizer, head = load_klev(a.ckpt, base=a.base, preset=a.preset, backend=a.backend)
    kwargs = {"temperature": a.temperature}
    if a.max_seq:
        kwargs["max_seq"] = a.max_seq
    STATE.update(model=model, tokenizer=tokenizer, head=head, lock=threading.Lock(),
                 name=a.name or a.ckpt, base=a.base, temperature=a.temperature,
                 garbage=bool(head.garbage),
                 answer=lambda req: answer(model, tokenizer, head, req, **kwargs))
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[serve] listening on http://{a.host}:{a.port}  (POST /v1/systemone, GET /v1/models)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped", flush=True)


if __name__ == "__main__":
    main()
