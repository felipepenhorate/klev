"""Encoder for distill_jev decision rows: Kev's format, Gemma tokenizer, row form.

One row = one question: `<bos><unused0> state <unused1> instruction <unused2> opt </opt> ... <|decide>`
(Kev packs every question of a record into one row with a block-causal mask; the row form is
exact on any architecture and is what the hybrid/vision backbones need, so every question is
its own row here, as in Kev's `forward_rows_batch`).

Positions are found by scanning the encoded ids for the delimiter token ids, which survives
the image expansion (`boi + n soft tokens + eoi`) without position arithmetic. The content
mask marks the positions whose *next* token is still state/instruction text: those are the
positions the knowledge-KL term anchors and, because option order changes never touch them,
they are exactly the prefix-teacher-cache key (3.4).
"""
from .config import DECIDE, OPTION, OPTION_END, QUESTION, STATE
from .format import state_text

_WANTED = ("state", "q", "opt", "opt_end", "decide")
_TOKENS = (STATE, QUESTION, OPTION, OPTION_END, DECIDE)


class ContextOverflow(ValueError):
    """A row does not encode within the training context (state or branch too long)."""


def escape_user(text: str) -> str:
    """Rewrite delimiter token strings in user text so they cannot be forged (Kev rewrites
    `<|name|>` to `<¦name¦>` before tokenizing, for the same reason)."""
    for token in _TOKENS:
        if token in text:
            text = text.replace(token, token.replace("<", "\u00a6").replace(">", "\u00a6"))
    return text


def delimiter_positions(tokenizer, ids) -> dict:
    """{name: [token indices]} for every delimiter occurrence, by id."""
    ids_wanted = {name: tokenizer.convert_tokens_to_ids(token) for name, token in zip(_WANTED, _TOKENS)}
    found = {name: [] for name in ids_wanted}
    for i, token_id in enumerate(ids):
        for name, wanted in ids_wanted.items():
            if token_id == wanted:
                found[name].append(i)
    return found


def build_row_text(state: str, instruction: str, options: list[str]) -> str:
    parts = [STATE, escape_user(state), QUESTION, escape_user(instruction)]
    for option in options:
        parts += [OPTION, escape_user(option), OPTION_END]
    parts.append(DECIDE)
    return "".join(parts)


def encode_row(tokenizer, rec, question, max_state: int = 384, max_branch: int = 1024,
               strict: bool = True, image_spans: list[tuple[int, int]] | None = None) -> dict:
    """Encode one (record, question) row -> ids, masks and readout positions.

    image_spans: [(start, end)] token ranges of image soft tokens, excluded from the content
    mask. The caller computes them while expanding `<|image|>` placeholders (M2)."""
    text = build_row_text(state_text(rec), question["instr"], question["options"])
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Gemma has an explicit `<bos>`; Qwen does not (bos_token_id is None), and
    # convert_tokens_to_ids("<bos>") returns None there rather than raising. Prepend the
    # tokenizer's own BOS only when it has one, and let the state-length arithmetic below
    # account for the difference.
    bos = tokenizer.bos_token_id
    ids = ([bos] + body) if bos is not None else list(body)
    pos = delimiter_positions(tokenizer, ids)
    n_options = len(question["options"])
    if not (len(pos["state"]) == 1 and len(pos["q"]) == 1 and len(pos["decide"]) == 1
            and len(pos["opt"]) == n_options and len(pos["opt_end"]) == n_options):
        raise AssertionError(f"row layout mismatch: { {k: len(v) for k, v in pos.items()} } for {n_options} options")
    if pos["decide"][0] != len(ids) - 1:
        raise AssertionError("the decide delimiter must be the last token of the row")
    first_opt = pos["opt"][0]
    # <bos> <STATE> state... <QUESTION>, so the state text is whatever sits between the two
    # delimiters, minus the leading BOS when the tokenizer has one.
    state_tokens = pos["q"][0] - 2 - (0 if bos is None else 1)
    branch_tokens = len(ids) - pos["q"][0]
    if strict and state_tokens > max_state:
        raise ContextOverflow(f"state too long: {state_tokens} > {max_state}")
    if strict and branch_tokens > max_branch:
        raise ContextOverflow(f"branch too long: {branch_tokens} > {max_branch}")

    content_mask = [0] * len(ids)
    for t in range(first_opt - 1):                      # t predicts t+1; the target must be prefix text
        content_mask[t] = 1
    for start, end in image_spans or []:
        for t in range(max(0, start - 1), min(end + 1, len(ids))):
            content_mask[t] = 0
    return {"input_ids": ids, "content_mask": content_mask,
            "opt_pos": list(pos["opt_end"]), "decide_pos": pos["decide"][0],
            "state_tokens": state_tokens, "branch_tokens": branch_tokens}


def build_target(question: dict, n_options: int) -> list[float]:
    """The K+1 pointer target: the option distribution plus the garbage mass.

    A hard label is one-hot over the K options, garbage 0; a soft `target` (K probabilities
    keyed by option order) is used as-is; a `garbage` mass g moves g of the total onto the
    rejection candidate and spreads the rest over the options."""
    if question.get("target") is not None:
        base = [float(x) for x in question["target"]]
        if len(base) != n_options:
            raise ValueError(f"target has {len(base)} entries for {n_options} options")
    else:
        base = [0.0] * n_options
        base[question["label"]] = 1.0
    g = float(question.get("garbage", 0.0) or 0.0)
    if not 0.0 <= g <= 1.0:
        raise ValueError("garbage mass must be in [0, 1]")
    return [x * (1.0 - g) for x in base] + [g]
