"""Verify the audited V4 operational core without storing a duplicate source tree.

The final repository replaces ``preserved_v4_snapshot/`` with a compact SHA256 manifest.
Every protected V4 source must match the audited digest. ``config.py`` has a separately
recorded V5.1 digest because its only accepted extension is environment-variable objective
overrides whose defaults remain the original V4 coefficients.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "provenance" / "v4_core_manifest.json"
LIVE = ROOT / "scripts"
REPORT_PATH = ROOT / "processed_data" / "v4_preservation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("files", [])
    if not expected:
        raise SystemExit("V4 preservation failed: empty core manifest")

    rows = []
    failures = []
    for item in expected:
        relative = Path(str(item["path"]))
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        wanted = str(item["sha256"])
        passed = actual == wanted
        rows.append(
            {
                "path": relative.as_posix(),
                "expected_sha256": wanted,
                "actual_sha256": actual,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(relative.as_posix())

    live_config = (LIVE / "config.py").read_text(encoding="utf-8")
    markers = [
        "MULTITAXI_DRIVER_REVENUE_SHARE",
        "MULTITAXI_OCCUPIED_COST_PER_MILE",
        "MULTITAXI_EMPTY_COST_PER_MILE",
        '"1.00"',
        '"0.35"',
    ]
    config_contract_passed = all(marker in live_config for marker in markers)
    if not config_contract_passed:
        failures.append("scripts/config.py:objective_override_contract")

    status = "V4 PRESERVATION PASSED" if not failures else "V4 PRESERVATION FAILED"
    report = {
        "status": status,
        "baseline_commit": manifest.get("baseline_commit"),
        "protected_files": len(expected),
        "config_contract_passed": config_contract_passed,
        "failed": failures,
        "files": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
