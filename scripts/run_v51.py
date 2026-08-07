"""Run the audited V4 core and the V5.1 accounting/attribution extensions."""
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
    exp = cfg.get("experiment", {})
    run = cfg.get("run", {})
    model = primary_cost_model(cfg)

    # The preserved V4 objective consumes revenue share and per-mile costs. A fixed
    # hourly lease is valid for post-hoc sensitivity, but cannot be the active policy
    # objective without changing V4. Fail instead of silently reporting a different
    # objective from the one used for training and control.
    if model.fixed_lease_per_hour > 1e-12:
        raise SystemExit(
            "Primary cost model cannot have fixed_lease_per_hour > 0 in the "
            "V5.1 pipeline. Use it only as a post-hoc sensitivity model."
        )

    env = os.environ.copy()
    env.update(model.env())
    env["PYTHONHASHSEED"] = str(exp.get("random_seed", 20260712))
    if run.get("deterministic_cpu", True):
        env["CUDA_VISIBLE_DEVICES"] = ""

    # Persist the exact active objective. cost_reaccounting.py requires these fields
    # and verifies that V4 episode accounting matches them before recovering receipts.
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
        }
    )

    import yaml

    resolved = ROOT / "processed_data" / "resolved_v51_config.yaml"
    resolved.parent.mkdir(exist_ok=True)
    resolved.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    call("verify_v4_preservation.py", env=env)
    call("verify_research_contract.py", "--static-only", env=env)
    call("preflight.py", "--config", str(resolved), env=env)

    if run.get("download_data", False) and not args.skip_download:
        call("download_data.py", env=env)
        call("verify_data.py", "--strict-lock", env=env)

    if (
        run.get("fare_share_calibration_check", True)
        and model.source_farebox_share_rate > 0.0
    ):
        call("calibrate_fare_share.py", "--config", str(resolved), env=env)

    if not args.skip_v4:
        v4 = [
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
            v4.extend(
                ["--include-q-learning", "--q-episodes", str(run.get("q_episodes", 10000))]
            )
        if run.get("include_dqn", True):
            v4.extend(
                ["--include-dqn", "--dqn-episodes", str(run.get("dqn_episodes", 8000))]
            )
        if run.get("include_legacy", False):
            v4.append("--include-legacy")
        if run.get("fast", False):
            v4.append("--fast")
        if run.get("skip_environment", False):
            v4.append("--skip-environment")
        if run.get("skip_validation", False):
            v4.append("--skip-validation")
        if run.get("skip_visualization", False):
            v4.append("--skip-visualization")
        if run.get("skip_competition_sensitivity", False):
            v4.append("--skip-competition-sensitivity")
        call("run_all.py", *v4, env=env)

    # run_all.py begins with the protected V4 clean_outputs.py, which intentionally
    # clears generated JSON reports. Recreate the integrity report after that cleanup so
    # final acceptance can validate the exact sources that produced the outputs.
    call("verify_v4_preservation.py", env=env)
    call("verify_research_contract.py", env=env)
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
    print("\nMULTITAXI-BENCH V5.1 PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
