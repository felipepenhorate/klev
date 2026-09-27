"""Frozen Kev suites: manifests in git, large partitions fetched from the Hub mirror and sha256-checked.

Semantics are Kev's (github.com/jaredpalmer/kev, kev/suite.py, Apache-2.0): a suite lives under
`evals/<version>/<suite>/` with `manifest.json` pinning every partition's sha256 and record count;
a partition missing locally is downloaded once from `jaredpalmer/kev-suites@SUITES_REVISION` (or the
manifest's own `mirror`) into place and verified against the manifest. Test partitions are locked
behind `allow_test=True`, which is read once per candidate.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"

SUITES_DATASET = "jaredpalmer/kev-suites"
SUITES_REVISION = "a88f56db5341397299137cb68775c2ea6e3f68cb"
SPLITS = ("train", "calibration", "development", "test")


def digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value, atomic: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    else:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_manifest(directory):
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (copy the suite's manifest into {Path(directory)})")
    return read_json(path)


def fetch_partition(directory, filename):
    """Download one partition of a frozen suite from its Hub mirror into place; the caller verifies
    the sha256."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    directory = Path(directory).resolve()
    if EVALS not in directory.parents:
        raise FileNotFoundError(f"{directory / filename} is missing and is not under {EVALS}")
    relative = directory.relative_to(EVALS) / filename
    mirror = read_manifest(directory).get("mirror")
    repo, revision = (mirror["dataset"], mirror["revision"]) if mirror else (SUITES_DATASET, SUITES_REVISION)
    try:
        cached = hf_hub_download(repo, str(relative), repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, GatedRepoError) as e:
        raise PermissionError(f"{relative} is only in {repo}, which is missing or private to this account; "
                              "`hf auth login` (or HF_TOKEN) with access to it, or ask for it") from e
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {repo}@{revision[:10]}", flush=True)


def load_split(directory, split, allow_test: bool = False):
    """One frozen partition as a list of records, verified against its manifest (sha256 + count)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit allow_test; never use it for search")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        fetch_partition(directory, path.name)
    if digest(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != manifest["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records
