"""Verify the V4 core plus the narrowly authorized v5.2 data repair.

The baseline SHA256 values remain recorded unchanged. A protected file may differ only
when ``v4_core_manifest.json`` names both its audited baseline digest and its exact
authorized replacement digest. This makes the date repair explicit instead of falsely
claiming byte-identical V4 preservation.
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
    approved = {
        str(item["path"]): item
        for item in manifest.get("authorized_changes", [])
    }
    if not expected:
        raise SystemExit("V4 preservation failed: empty core manifest")
    if not approved:
        raise SystemExit("V5.2 preservation failed: no authorized repair manifest")

    rows = []
    failures = []
    baseline_paths = {str(item["path"]) for item in expected}
    for item in expected:
        relative = Path(str(item["path"]))
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        baseline = str(item["sha256"])
        repair = approved.get(relative.as_posix())
        wanted = (
            str(repair["authorized_sha256"])
            if repair is not None
            else baseline
        )
        passed = actual == wanted
        rows.append(
            {
                "path": relative.as_posix(),
                "baseline_sha256": baseline,
                "authorized_sha256": wanted,
                "actual_sha256": actual,
                "verification_mode": (
                    "AUTHORIZED_V5.2_DATE_FIX"
                    if repair is not None
                    else "BYTE_IDENTICAL_V4"
                ),
                "passed": passed,
            }
        )
        if not passed:
            failures.append(relative.as_posix())

    # Verify newly added repair helpers that have no V4 baseline entry.
    for relative_text, repair in approved.items():
        if relative_text in baseline_paths:
            continue
        relative = Path(relative_text)
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        wanted = str(repair["authorized_sha256"])
        passed = actual == wanted
        rows.append(
            {
                "path": relative.as_posix(),
                "baseline_sha256": None,
                "authorized_sha256": wanted,
                "actual_sha256": actual,
                "verification_mode": "AUTHORIZED_V5.2_ADDITION",
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

    status = (
        "V4 CORE WITH AUTHORIZED V5.2 DATE FIX PASSED"
        if not failures
        else "V4 CORE WITH AUTHORIZED V5.2 DATE FIX FAILED"
    )
    report = {
        "status": status,
        "baseline_branch": manifest.get("baseline_branch"),
        "baseline_commit": manifest.get("baseline_commit"),
        "project_version": "5.2.0",
        "protected_v4_files": len(expected),
        "authorized_changes": sorted(approved),
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
