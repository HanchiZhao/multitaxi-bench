"""Run MultiTaxi-Bench v5.2 with the authorized pickup-month repair."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from cost_models import load_yaml, primary_cost_model


ROOT = Path(__file__).resolve().parents[1]


def call(script: str, *args: object, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    print("\n" + "=" * 96)
    print("RUN:", " ".join(cmd))
    print("=" * 96)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-v4", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if str(cfg.get("project_version")) != "5.2.0":
        raise SystemExit("v5.2 runner requires project_version: 5.2.0")
    exp = cfg.get("experiment", {})
    run = cfg.get("run", {})
    model = primary_cost_model(cfg)

    if model.fixed_lease_per_hour > 1e-12:
        raise SystemExit(
            "Primary cost model cannot have fixed_lease_per_hour > 0. "
            "Use it only as a post-hoc sensitivity model."
        )

    env = os.environ.copy()
    env.update(model.env())
    env["PYTHONHASHSEED"] = str(exp.get("random_seed", 20260712))
    if run.get("deterministic_cpu", True):
        env["CUDA_VISIBLE_DEVICES"] = ""

    cfg.setdefault("runtime", {}).update(
        {
            "active_cost_model": model.name,
            "active_revenue_share": model.revenue_share,
            "active_occupied_cost_per_mile": model.occupied_cost_per_mile,
            "active_empty_cost_per_mile": model.empty_cost_per_mile,
            "active_fixed_lease_per_hour": model.fixed_lease_per_hour,
            "download_requested_by_config": bool(run.get("download_data", False)),
            "skip_download_cli": bool(args.skip_download),
            "effective_download_performed": bool(
                run.get("download_data", False) and not args.skip_download
            ),
            "skip_v4_cli": bool(args.skip_v4),
            "date_boundary_fix": (
                "declared month start <= pickup < next month start; "
                "dropoff is not month-restricted"
            ),
            "environment_schema_version": "4.1-date-boundary-fix",
        }
    )

    import yaml

    resolved = ROOT / "processed_data" / "resolved_v52_config.yaml"
    resolved.parent.mkdir(exist_ok=True)
    resolved.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    call("verify_v4_preservation.py", env=env)
    call("verify_research_contract.py", "--static-only", env=env)
    call("preflight.py", "--config", str(resolved), env=env)

    if run.get("download_data", False) and not args.skip_download:
        call("download_data.py", env=env)
    if run.get("strict_data_lock", False):
        call("verify_data.py", "--strict-lock", env=env)

    if (
        run.get("fare_share_calibration_check", True)
        and model.source_farebox_share_rate > 0.0
    ):
        call("calibrate_fare_share.py", "--config", str(resolved), env=env)

    if not args.skip_v4:
        pipeline_args = [
            "--start-zone",
            str(exp.get("start_zone", 132)),
            "--start-time",
            str(exp.get("start_time", "08:00")),
            "--episodes",
            str(exp.get("episodes", 300)),
            "--players",
            str(exp.get("global_shapley_players", 10)),
            "--permutations",
            str(exp.get("global_shapley_permutations", 512)),
            "--competition-scenario",
            str(exp.get("competition_scenario", "medium")),
        ]
        if run.get("include_q_learning", True):
            pipeline_args.extend(
                ["--include-q-learning", "--q-episodes", str(run.get("q_episodes", 10000))]
            )
        if run.get("include_dqn", True):
            pipeline_args.extend(
                ["--include-dqn", "--dqn-episodes", str(run.get("dqn_episodes", 8000))]
            )
        if run.get("include_legacy", False):
            pipeline_args.append("--include-legacy")
        if run.get("fast", False):
            pipeline_args.append("--fast")
        if run.get("skip_environment", False):
            pipeline_args.append("--skip-environment")
        if run.get("skip_validation", False):
            pipeline_args.append("--skip-validation")
        if run.get("skip_visualization", False):
            pipeline_args.append("--skip-visualization")
        if run.get("skip_competition_sensitivity", False):
            pipeline_args.append("--skip-competition-sensitivity")
        call("run_all.py", *pipeline_args, env=env)

    # run_all starts with the protected cleanup stage. Recreate integrity reports after
    # it completes so acceptance validates the exact final output set.
    call("verify_v4_preservation.py", env=env)
    call("verify_research_contract.py", env=env)
    if run.get("strict_data_lock", False):
        call("verify_data.py", "--strict-lock", env=env)
        call("verify_date_boundary_fix.py", env=env)
    call("cost_reaccounting.py", "--config", str(resolved), env=env)
    call(
        "path_shapley_explainer.py",
        "--config",
        str(resolved),
        "--permutations",
        str(cfg.get("path_shapley", {}).get("permutations", 512)),
        env=env,
    )
    call("path_axiom_validation.py", "--config", str(resolved), env=env)
    call("visualize_path_shapley.py", "--config", str(resolved), env=env)
    call("verify_reproduction.py", "--config", str(resolved), env=env)
    print("\nMULTITAXI-BENCH V5.2 PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
