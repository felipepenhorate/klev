"""TypeSafe-compatible request/response format and the internal record builder.

Vendored from kev/api.py and kev/data.py (Apache-2.0, github.com/jaredpalmer/kev) so the
decision format is identical to Kev's, plus a multimodal `state` extension:

    {"state": [{"type": "text", "text": "Look at the chart."},
               {"type": "image", "base64": "data:image/png;base64,..."}],
     "questions": {"trend": {"type": "noul", "instructions": "...", "label": true}}}

A plain JSONContent state (str/dict/list) is rendered by kev's `render()` and produces a
text-only record exactly as Kev does. A state list containing image parts produces
`rec["state"] = {"text": ..., "images": [<base64>, ...]}` instead of a string.
"""
import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "distill_jev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria names
    (choice), ["false", "true"] (noul), the level indices as strings (score). Labels, targets and
    anchors use the same keys."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: SystemOneRequest):
    """-> internal record for encode(), plus per-question metadata ({"id", "type", "keys", "legend" for
    score}) to map probabilities back."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q.instructions), "options": opts, "label": 0})
        meta.append(m)
    return {"state": render(req.state), "questions": qs}, meta


# --- images ------------------------------------------------------------------------------------------------------------

IMAGE_TYPES = ("image", "image_url")


def strip_data_uri(value: str) -> str:
    """Accept a data URI or a bare base64 string; return the base64 payload."""
    if not isinstance(value, str) or not value:
        raise ValueError("an image part needs base64 data")
    if value.startswith("data:"):
        head, sep, payload = value.partition(",")
        if not sep:
            raise ValueError("malformed data URI")
        return payload
    return value


def data_uri(b64: str, media_type: str = "image/png") -> str:
    return b64 if b64.startswith("data:") else f"data:{media_type};base64,{b64}"


def _image_value(part: dict) -> str:
    for key in ("base64", "image", "image_url", "url", "data"):
        value = part.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and isinstance(value.get("url"), str):  # {"image_url": {"url": ...}}
            return value["url"]
    raise ValueError("image part has no base64/url field")


def split_state(state: JSONContent) -> tuple[str, list[str]]:
    """-> (text the model sees, list of base64 images). Image content parts are pulled out; every
    other part is rendered as text. A plain JSONContent state renders to `render(state)`."""
    if isinstance(state, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_TYPES for p in state):
        texts, images = [], []
        for part in state:
            if isinstance(part, dict) and part.get("type") in IMAGE_TYPES:
                images.append(strip_data_uri(_image_value(part)))
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(render(part.get("text")))
            else:
                texts.append(render(part))
        return "\n".join(t for t in texts if t), images
    return render(state), []


def state_text(rec) -> str:
    state = rec["state"]
    return state["text"] if isinstance(state, dict) else state


def state_images(rec) -> list[str]:
    state = rec["state"]
    return state.get("images", []) if isinstance(state, dict) else []


def api_request(record):
    """The /v1/systemone request body for a labelled record: state and typed questions only, never
    labels, targets or metadata."""
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()}}


def materialize(req):
    """Labelled request -> internal record via the serving path (api.to_record), attaching int labels
    and src. Image content parts become `{"text", "images"}` state; text-only states stay strings,
    byte-identical to Kev's records."""
    text, images = split_state(req["state"])
    stripped = {"state": text or "", "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in req["questions"].items()}}
    rec, meta = to_record(SystemOneRequest.model_validate(stripped))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        if "label" in src_q:   # a garbage question needs no option label
            y = src_q["label"]
            q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)
        else:
            q["label"] = 0
        q["src"] = src_q.get("src", ""); q["qtype"] = m["type"]; q["qid"] = qid; q["keys"] = m["keys"]
        q["garbage"] = float(src_q.get("garbage", 0.0) or 0.0)
        if src_q.get("target") is not None:
            t = [float(src_q["target"].get(k, 0.0)) for k in q["keys"]]
            if sum(t) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
    if images:
        rec["state"] = {"text": text, "images": images}
    return rec


def choice_confidence(p: list[float]) -> float:
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p: list[float]) -> float:
    """Approximation of TypeSafe's 'distance from the modal level' statistic."""
    L = len(p); mode = max(range(L), key=lambda i: p[i])
    return 1.0 if L == 1 else 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def round_prob(x: float) -> float:
    return round(float(x), 4)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": round_prob(choice_confidence(p)), "probabilities": dist}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": round_prob(score), "legend": m["legend"],
                            "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                            "confidence": round_prob(score_confidence(p))}
    return out
