"""Run the complete dynamic two-hour pipeline in the required order."""
from __future__ import annotations

import argparse
import subprocess
import sys

from config import (
    DEFAULT_COMPETITION_SCENARIO, DQN_EPISODES, PROJECT_ROOT,
    Q_LEARNING_EPISODES,
)


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *map(str, args)]
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MultiTaxi dynamic two-hour analyses")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--players", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--competition-scenario", choices=["low", "medium", "high"], default=DEFAULT_COMPETITION_SCENARIO)
    parser.add_argument("--include-q-learning", action="store_true")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--include-dqn", action="store_true", help="Train and evaluate the dynamic-action Double DQN")
    parser.add_argument("--q-episodes", type=int, default=None)
    parser.add_argument("--dqn-episodes", type=int, default=None)
    parser.add_argument("--skip-environment", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--skip-competition-sensitivity", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true", help="Do not clear old generated CSV/figures before rebuilding the environment")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    episodes = min(args.episodes, 50) if args.fast else args.episodes
    permutations = min(args.permutations, 96) if args.fast else args.permutations
    players = min(args.players, 6) if args.fast else args.players

    if not args.skip_environment:
        if not args.keep_outputs:
            run("clean_outputs.py")
        run("route_environment.py", "--competition-scenario", args.competition_scenario)
        run("test_dynamic_pipeline.py")

    run("validate_full_od_coverage.py", "--sample", "300" if args.fast else "5000")
    run("scenario_bank.py", "--count", str(episodes))

    if args.include_q_learning:
        q_episodes = str(args.q_episodes or (1000 if args.fast else Q_LEARNING_EPISODES))
        run("train_policies.py", "--start-zone", str(args.start_zone), "--start-time", args.start_time, "--episodes", q_episodes)
    if args.include_dqn:
        dqn_episodes = str(args.dqn_episodes or (1500 if args.fast else DQN_EPISODES))
        run(
            "train_dqn.py", "--start-zone", str(args.start_zone), "--start-time", args.start_time,
            "--episodes", dqn_episodes, "--competition-scenario", args.competition_scenario,
            "--validation-episodes", "30" if args.fast else "120",
            "--validation-interval", "250",
        )

    eval_args = [
        "--start-zone", str(args.start_zone), "--start-time", args.start_time,
        "--episodes", str(episodes), "--competition-scenario", args.competition_scenario,
    ]
    if args.include_q_learning:
        eval_args.append("--include-q-learning")
    if args.include_legacy:
        eval_args.append("--include-legacy")
    if args.include_dqn:
        eval_args.append("--include-dqn")
    run("evaluate_recommendations.py", *eval_args)

    run(
        "shapley_explainer.py",
        "--start-zone", str(args.start_zone), "--start-time", args.start_time,
        "--players", str(players), "--permutations", str(permutations),
        "--competition-scenario", args.competition_scenario,
        "--calibration-episodes", "30" if args.fast else "150",
    )

    if not args.skip_validation:
        run("imputation_validation.py", "--sample", "400" if args.fast else "2000")
        run("ablation_study.py", "--top-k", "3" if args.fast else "5", "--random-controls", "4" if args.fast else "20")
        run("baseline_sensitivity.py", "--permutations", str(permutations))
        if args.fast:
            run("axiom_validation.py", "--skip-dim", "--skip-dme")
        else:
            run("axiom_validation.py")
        if not args.skip_competition_sensitivity:
            sensitivity_args = [
                "--start-zone", str(args.start_zone), "--start-time", args.start_time,
                "--episodes", "30" if args.fast else "150",
            ]
            if args.include_q_learning:
                sensitivity_args.append("--include-q-learning")
            if args.include_dqn:
                sensitivity_args.append("--include-dqn")
            if args.include_legacy:
                sensitivity_args.append("--include-legacy")
            run("competition_sensitivity.py", *sensitivity_args)

    if not args.skip_visualization:
        run("visualize_results.py")
    print("\nAll requested dynamic two-hour stages completed successfully.")


if __name__ == "__main__":
    main()
