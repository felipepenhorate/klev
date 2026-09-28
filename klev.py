"""klev -- a typed decision model: calibrated option probabilities plus a rejection channel.

This is the public entry point. `pip install klev`, then:

    from klev import load_klev, answer

    model, tokenizer, head = load_klev("lumierenoir/klev-0.8b")
    answer(model, tokenizer, head, {"state": "...", "questions": {...}}, temperature=2.2974)

and for a service:

    klev-serve --ckpt lumierenoir/klev-0.8b --port 8090

The implementation lives in `model/`, `data/`, `serve/` and `examples/` at the repo root, which
this package makes importable. The modules use top-level absolute imports (`from data.config
import ...`), which is how the research code and every eval script have always referred to each
other, so installing that root on `sys.path` is what keeps one implementation rather than two.

The cost is that `data`, `model` and `eval` are generic enough to collide with another
distribution's top-level names. The alternative is a full src-layout rename of every import in
the repo, which would break every script and eval that currently works; if that trade ever
needs revisiting it is a single deliberate change, not an ongoing cost. `MODEL` below re-exports
the same name as `data.config.MODEL` for callers that want the current base without a loader.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.config import MODEL, PRESETS  # noqa: E402
from model.infer import answer, load_klev  # noqa: E402

__version__ = "0.1.0"
__all__ = ["load_klev", "answer", "serve", "system_one", "MODEL", "PRESETS", "__version__"]


def serve():
    """Entry point for `klev-serve` -- the optional TypeSafe-compatible HTTP endpoint.

    Both mains take no arguments: they parse sys.argv themselves, which is what a console
    script wants. The wrappers exist only to make the import of the repo-root packages happen
    first (klev.py puts its own directory on sys.path on import)."""
    from serve.server import main
    return main()


def system_one():
    """Entry point for `klev-system-one` -- decisions from a shell, no server."""
    from examples.system_one import main
    return main()
