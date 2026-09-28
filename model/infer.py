"""Shared inference: base + adapter + head.pt -> typed answers for a SystemOne request.

The one place that turns a decision request into probabilities. `eval/eval_decisions.py`
scores *prepared* rows, which is right for a benchmark and wrong for a user: preparing a row
means running data/prep_dataset.py and a `datasets` directory. This module skips all that --
hand it a `{"state", "questions"}` dict and get typed answers back.

It is the same arithmetic as the scorer on purpose: same delimiter registration, same
`encode_row` positions, same pointer readout, same renormalisation over the K options with the
garbage candidate reported as `none`. If this drifts from eval/eval_decisions.py the published
numbers stop describing the model, so the two share data/encoding.py, data/format.py and
model/head.py rather than reimplementing any of it.

    from model.infer import load_klev, answer
    model, tokenizer, head = load_klev(ckpt="lumierenoir/klev-0.8b")
    print(answer(model, tokenizer, head, {
        "state": "My card was declined at a shop.",
        "questions": {"can_i_help": {"type": "noul", "instructions": "Is this a card problem?",
                                     "criteria": {"false": "not a card problem",
                                                  "true": "a card problem"}}}}))
"""
import json
from pathlib import Path

import torch

from data import config
from data.encoding import ContextOverflow, encode_row
from data.format import SystemOneRequest, to_answers, to_record
from model.delimiters import apply_delimiter_deltas
from model.head import PointerHead
from model.load import compute_dtype, language_model


def load_klev(ckpt, base=None, preset="", device="cuda", max_seq_length=2048, cache_dir=None):
    """Load a published (or local) klev checkpoint: base, LoRA adapter, pointer head, deltas.

    `ckpt` is either a local directory holding `adapter/` + `head.pt`, or a Hub repo id, which is
    downloaded into `cache_dir` (default: the HF cache). This is the whole load path -- there is
    no server and no other setup step.
    """
    import unsloth  # noqa: F401  (must precede transformers/peft; patches them at import)
    from peft import PeftModel
    from unsloth import FastModel

    config.apply_preset(preset)
    base = base or config.MODEL
    run = _resolve(ckpt, cache_dir)

    model, processor = FastModel.from_pretrained(
        model_name=base, max_seq_length=max_seq_length, dtype=compute_dtype(),
        load_in_4bit=True, use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": config.DELIMITERS})
    deltas = apply_delimiter_deltas(model, language_model,
                                    [tokenizer.convert_tokens_to_ids(t) for t in config.DELIMITERS])
    model = PeftModel.from_pretrained(model, str(run / "adapter"))

    saved = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    for name, tensor in saved["deltas"].items():
        if name in deltas:
            deltas[name].data.copy_(tensor.to(deltas[name].device))
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size")
    head = PointerHead(hidden_size, dp=saved["head"]["q.weight"].shape[0],
                       garbage=bool(saved.get("garbage", False)))
    head.load_state_dict(saved["head"])
    # head.pt stores 1.0 because it is written during training (the head only tempers in eval
    # mode). Pass temperature= to answer() to serve a calibrated distribution.
    head.temperature = 1.0
    model.head = head.to(device).eval()
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model, tokenizer, head


def _resolve(ckpt, cache_dir=None):
    """A local dir, or a Hub repo id fetched for `adapter/` + `head.pt`."""
    path = Path(str(ckpt))
    if (path / "head.pt").exists():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(ckpt), cache_dir=cache_dir,
                                  allow_patterns=["adapter/*", "head.pt"]))


@torch.no_grad()
def answer(model, tokenizer, head, request, temperature=1.0, max_state=None, max_branch=None,
           max_seq=None, device=None):
    """One SystemOne request -> {"answers": {qid: typed answer}}.

    `request` is `{"state": ..., "questions": {qid: {"type", "instructions", "criteria"}}}` --
    the same body the HTTP endpoint takes, and what data/format.py's api_request() emits.
    Unknown keys (labels, targets, _meta) are ignored, so a labelled eval record can be passed
    straight through.

    `temperature` divides the pointer logits before the softmax (PointerHead._temper). 1.0 is
    the raw readout, which is what eval/eval_decisions.py scores. The values fitted by
    scripts/calibrate.py -- 2.2974 for the IT arm, 2.2449 for -Base -- are what produce the
    served numbers in the model card; a head fitted per model is what you want in production.
    """
    head.temperature = float(temperature)
    device = device or next(model.parameters()).device
    max_state = max_state or config.MAX_STATE
    max_branch = max_branch or config.MAX_BRANCH
    max_seq = max_seq or config.MAX_SEQ

    body = {"state": request["state"],
            "questions": {q: {k: v for k, v in spec.items()
                              if k in ("type", "instructions", "criteria")}
                          for q, spec in request["questions"].items()}}
    rec, meta = to_record(SystemOneRequest.model_validate(body))

    answers, probs_all = {}, []
    for question, m in zip(rec["questions"], meta):
        try:
            enc = encode_row(tokenizer, rec, question, max_state=max_state, max_branch=max_branch)
        except ContextOverflow as e:
            raise ValueError(f"question {m['id']!r} does not fit: {e}") from e
        ids = torch.tensor([enc["input_ids"]], device=device)
        if ids.shape[1] > max_seq:
            raise ValueError(f"question {m['id']!r} is {ids.shape[1]} tokens, over --max-seq {max_seq}")
        hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       output_hidden_states=True, use_cache=False,
                       logits_to_keep=1).hidden_states[-1][0]
        options = hidden[torch.tensor(enc["opt_pos"], device=device)]
        decide = hidden[enc["decide_pos"]]
        # K+1 logits: the K options plus the learned garbage (rejection) candidate.
        logits = model.head(decide.float(), options.float())
        probs = torch.softmax(logits.float(), -1)
        probs_all.append(probs.tolist())

    typed = to_answers(probs_all, meta)
    for m, probs, out in zip(meta, probs_all, typed.values()):
        # The readout returns K+1 logits, so the rejection candidate sits at index len(keys) for
        # all three question types (choice -> K criteria, noul -> 2, score -> L levels). The K
        # options are renormalised by 1 - p_none inside to_answers' caller-side convention, so
        # the returned distribution sums to 1 and `none` is the rejection channel. A kev-format
        # head has no garbage candidate, hence the 0.0 branch.
        out["none"] = round(float(probs[len(m["keys"])]), 6) if head.garbage else 0.0
    return {"answers": typed}


def answer_json(model, tokenizer, head, request_json, **kwargs):
    """Same as answer() but takes and returns JSON text, for the HTTP layer."""
    return json.dumps(answer(model, tokenizer, head, json.loads(request_json), **kwargs))
