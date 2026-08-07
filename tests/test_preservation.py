import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "v4_core_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audited_v4_core_manifest() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["baseline_commit"] == "a212a445478dde7037ed12de539eb19a9c3f35fe"
    assert len(payload["files"]) == 23
    for item in payload["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256(path) == item["sha256"], item["path"]


def test_no_duplicate_v4_source_snapshot() -> None:
    assert not (ROOT / "preserved_v4_snapshot").exists()
