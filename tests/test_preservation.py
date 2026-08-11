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
    approved = {item["path"]: item for item in payload["authorized_changes"]}
    assert set(approved) == {
        "scripts/route_environment.py",
        "scripts/month_boundaries.py",
    }
    for item in payload["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        expected = approved.get(item["path"], {}).get(
            "authorized_sha256", item["sha256"]
        )
        assert sha256(path) == expected, item["path"]
    for relative, item in approved.items():
        if any(row["path"] == relative for row in payload["files"]):
            continue
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == item["authorized_sha256"], relative


def test_no_duplicate_v4_source_snapshot() -> None:
    assert not (ROOT / "preserved_v4_snapshot").exists()
