"""Verify the committed MultiTaxi-Bench v5.2 evidence set."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "paper_artifacts" / "final_20260811"
MANIFEST = ARTIFACT_ROOT / "SHA256SUMS.csv"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
EXPECTED_ALGORITHMS = {
    "wait_only",
    "highest_demand",
    "highest_income",
    "greedy_net_earnings",
    "finite_horizon_value_iteration",
    "q_learning",
    "dqn",
}


def canonical_bytes(path: Path) -> bytes:
    """Return cross-platform bytes for release hashing.

    Git may check text artifacts out with LF or CRLF line endings.  The release
    manifest therefore hashes text after normalizing line endings to LF, while
    binary figures remain byte-exact.
    """
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def canonical_size(path: Path) -> int:
    return len(canonical_bytes(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Release manifest not found: {MANIFEST}")

    failures: list[str] = []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = {str(row["relative_path"]) for row in rows}
    actual_paths = {
        path.relative_to(ARTIFACT_ROOT).as_posix()
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        failures.append(f"manifest file-set mismatch: missing={missing}, unexpected={unexpected}")

    for row in rows:
        relative = str(row["relative_path"])
        path = ARTIFACT_ROOT / relative
        if not path.is_file():
            continue
        expected_size = int(row["size_bytes"])
        if canonical_size(path) != expected_size:
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != str(row["sha256"]):
            failures.append(f"SHA256 mismatch: {relative}")

    acceptance = json.loads(
        (ARTIFACT_ROOT / "reports" / "final_experiment_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    if acceptance.get("status") != "FINAL EXPERIMENT ACCEPTANCE PASSED":
        failures.append("final acceptance status is not PASSED")
    if acceptance.get("failed"):
        failures.append(f"final acceptance failed list is not empty: {acceptance['failed']}")

    environment = pd.read_csv(
        ARTIFACT_ROOT / "tables" / "environment_build_summary.csv"
    ).iloc[0]
    if (
        int(environment["service_days"]) != 181
        or int(environment["out_of_month_clean_rows_excluded"]) != 134
        or int(environment["retained_out_of_month_rows"]) != 0
    ):
        failures.append("date boundary is not 181 / 134 / 0")

    episodes = pd.read_csv(
        ARTIFACT_ROOT / "tables" / "policy_episode_results.csv",
        usecols=["algorithm", "scenario_id", "start_zone", "start_time", "horizon_minutes"],
    )
    algorithms = set(episodes["algorithm"].astype(str))
    counts = episodes.groupby("algorithm")["scenario_id"].size()
    scenario_sets = [
        set(group["scenario_id"].tolist())
        for _, group in episodes.groupby("algorithm", sort=False)
    ]
    if algorithms != EXPECTED_ALGORITHMS:
        failures.append(f"unexpected policy set: {sorted(algorithms)}")
    if len(episodes) != 2100 or not bool((counts == 300).all()):
        failures.append(f"evaluation size is not 7 x 300: rows={len(episodes)}, counts={counts.to_dict()}")
    if not scenario_sets or any(values != scenario_sets[0] for values in scenario_sets[1:]):
        failures.append("policies do not share the same scenario identifiers")
    if set(episodes["start_zone"].astype(int)) != {132}:
        failures.append("evaluation start zone is not uniquely 132")
    if set(episodes["start_time"].astype(str)) != {"08:00"}:
        failures.append("evaluation start time is not uniquely 08:00")
    if set(episodes["horizon_minutes"].astype(int)) != {120}:
        failures.append("evaluation horizon is not uniquely 120 minutes")

    if failures:
        for failure in failures:
            print("FAIL:", failure)
        raise SystemExit(f"V5.2 RELEASE ARTIFACT VERIFICATION FAILED ({len(failures)} issues)")

    print(f"Verified {len(rows)} files in {ARTIFACT_ROOT.relative_to(ROOT)}")
    print("FINAL EXPERIMENT ACCEPTANCE PASSED | failed = []")
    print("DATE BOUNDARY 181 / 134 / 0")
    print("EVALUATION 7 x 300 = 2100 COMMON-SCENARIO EPISODES")
    print("V5.2 RELEASE ARTIFACTS VERIFIED")


if __name__ == "__main__":
    main()
