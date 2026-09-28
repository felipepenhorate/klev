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


def load_klev(ckpt, base=None, preset="", device="cuda", max_seq_length=2048, cache_dir=None,
              backend=None):
    """Load a published (or local) klev checkpoint: base, LoRA adapter, pointer head, deltas.

    `ckpt` is either a local directory holding `adapter/` + `head.pt`, or a Hub repo id, which is
    downloaded into `cache_dir` (default: the HF cache). This is the whole load path -- there is
    no server and no other setup step.

    `backend` picks how the base is loaded: "unsloth" (default, `FastModel`, the stack training
    used) or "plain" (transformers + bitsandbytes, no unsloth). Unsloth exists here to make the
    *4-bit fine-tune* fit one consumer GPU; a prediction is one forward pass, and plain
    transformers runs it from the same checkpoint -- see docs/m15-plain-inference.md for the
    parity numbers. `KLEV_BACKEND=plain` sets the default.
    """
    from peft import PeftModel

    backend = (backend or os.environ.get("KLEV_BACKEND") or "unsloth").lower()
    if backend not in ("unsloth", "plain"):
        raise SystemExit(f"unknown backend {backend!r}; expected 'unsloth' or 'plain'")

    run = _resolve(ckpt, cache_dir)
    if base is None:
        # Take the base from the checkpoint rather than from the global default. A published
        # adapter records its own base in adapter_config.json, and pairing one with another
        # model's weights is silent nonsense -- the readout would just be wrong, with no error.
        # The delimiters then have to follow, so the base selects the preset.
        base = _base_of(run)
        if base is not None:
            matched = _preset_for(base)
            if matched and not preset:
                preset = matched
            elif not matched and not preset:
                raise SystemExit(
                    f"{run} is based on {base!r}, which matches no preset, so its delimiters are "
                    f"unknown. Pass preset= explicitly. Known bases: "
                    + ", ".join(f"{k} -> {v['model']}" for k, v in sorted(config.PRESETS.items())))
    config.apply_preset(preset)
    base = base or config.MODEL

    if backend == "unsloth":
        model, processor = _load_base_unsloth(base, max_seq_length)
    else:
        model, processor = _load_base_plain(base)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": config.DELIMITERS})
    deltas = apply_delimiter_deltas(model, language_model,
                                    [tokenizer.convert_tokens_to_ids(t) for t in config.DELIMITERS])
    if backend == "unsloth":
        model = PeftModel.from_pretrained(model, str(run / "adapter"))
    else:
        model = _attach_adapter_plain(model, Path(run) / "adapter")

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


def _base_of(run):
    """The base model a checkpoint's adapter was trained on, or None if it does not say."""
    path = Path(run) / "adapter" / "adapter_config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except Exception:
        return None


def _preset_for(base):
    """Which preset uses this base. The delimiters are the reason this matters: a klev trained on
    Qwen reads `<|fim_prefix|>`-style rows and one trained on Gemma reads `<unused0>`.., and
    feeding a model the wrong set fails as a bare
    `AssertionError: row layout mismatch: {'state': 0, 'q': 0, ...}` with every count zero."""
    base = str(base)
    for name, spec in config.PRESETS.items():
        if base == spec["model"]:
            return name
    # HF ids are sometimes stored with or without an `unsloth/` prefix or a revision suffix
    tail = base.rsplit("/", 1)[-1]
    for name, spec in config.PRESETS.items():
        if tail and tail == str(spec["model"]).rsplit("/", 1)[-1]:
            return name
    return None


def _load_base_unsloth(base, max_seq_length):
    """The training stack: unsloth's FastModel, which patches the Gemma 4 / Qwen 3.5 forward."""
    import unsloth  # noqa: F401  (must precede transformers/peft; patches them at import)
    from unsloth import FastModel

    return FastModel.from_pretrained(
        model_name=base, max_seq_length=max_seq_length, dtype=compute_dtype(),
        load_in_4bit=True, use_gradient_checkpointing=False, trust_remote_code=False)


def _load_base_plain(base):
    """The same 4-bit checkpoint through plain transformers + bitsandbytes, no unsloth.

    Everything klev needs at prediction time is stock: a causal/conditional-generation forward,
    bitsandbytes NF4 weights, and peft for the LoRA. The class is picked from the config so both
    presets work -- `Gemma4ForConditionalGeneration` (multimodal wrapper) for the Gemma arms,
    `...ForCausalLM` for the Qwen one. `max_seq_length` is unsloth-only: it configures RoPE
    scaling, which nothing here needs at the <=2k sequences a decision row is capped at.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, \
        AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    dtype = compute_dtype()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    try:
        # The processor carries the vision stack, which a decision row never touches. Gemma 4
        # pulls Pillow in for it, so fall back to the tokenizer alone when that is not installed.
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=False)
    except (ImportError, ValueError):
        processor = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    arch = (getattr(AutoConfig.from_pretrained(base, trust_remote_code=False), "architectures",
                    None) or [""])[0]
    auto_cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    kwargs = dict(quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa",
                  trust_remote_code=False)
    try:
        model = auto_cls.from_pretrained(base, dtype=dtype, **kwargs)
    except TypeError:                      # transformers < 5 spelling
        model = auto_cls.from_pretrained(base, torch_dtype=dtype, **kwargs)
    return model, processor


def _attach_adapter_plain(model, adapter_dir):
    """peft's LoRA injection, scoped to the text stack.

    The published adapter's `target_modules` are the usual projection names, and unsloth's
    training-time model applied them everywhere those names appear -- including Gemma 4's vision
    and audio towers, whose projections are `Gemma4ClippableLinear` wrappers. peft refuses those
    (it wraps `nn.Linear`, bnb `Linear4bit` and a handful of others), so injecting by name dies on
    the vision tower. The wrapper is unwrappable in principle but pointless here: a decision row
    is text, so the tower LoRA is never executed, and the text stack is plain `Linear4bit`, which
    peft handles natively. So the plain backend pins the targets to the language-model modules and
    leaves the vision/audio weights unused, which is what a text-only deployment does anyway.
    """
    from peft import LoraConfig, PeftModel

    cfg = LoraConfig.from_pretrained(str(adapter_dir))
    names = set(cfg.target_modules or [])
    keys = sorted(name for name, module in model.named_modules()
                  if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
                  and name.rsplit(".", 1)[-1] in names
                  and not _is_auxiliary_tower(name))
    if not keys:
        raise SystemExit(f"no text-stack module matches the adapter targets {sorted(names)}")
    cfg.target_modules = keys
    return PeftModel.from_pretrained(model, str(adapter_dir), config=cfg)


def _is_auxiliary_tower(name):
    """True for the vision/audio stacks of a multimodal checkpoint (never executed on text)."""
    return any(part in name for part in ("vision_tower", "audio_tower", "vision_model", "mm_projector"))


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
