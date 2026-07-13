"""Sensitivity of policy performance to latent vacant-taxi competition assumptions."""
from __future__ import annotations

import argparse
import pandas as pd

from config import HORIZON_MINUTES, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, SCENARIO_COUNT, ensure_directories
from evaluate_recommendations import episode_to_row, summarize_episode_results
from policies import QLearningPolicy, make_core_policies
from scenario_bank import build_scenario_bank
from two_hour_environment import DynamicTaxiEnvironment, parse_clock_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate low/medium/high competition scenarios")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--candidate-zones", type=int, default=12)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--include-q-learning", action="store_true")
    parser.add_argument("--include-dqn", action="store_true")
    args = parser.parse_args()
    ensure_directories()

    base_env = DynamicTaxiEnvironment.load()
    scenario_path = PROCESSED_DIR / "scenario_bank.csv"
    if scenario_path.exists():
        scenarios = pd.read_csv(scenario_path).head(args.episodes)
    else:
        scenarios = build_scenario_bank(max(args.episodes, SCENARIO_COUNT), RANDOM_SEED).head(args.episodes)
    if len(scenarios) < args.episodes:
        scenarios = build_scenario_bank(args.episodes, RANDOM_SEED)

    q_policy = None
    if args.include_q_learning and (MODELS_DIR / "q_learning_policy.pkl").exists():
        q_policy = QLearningPolicy.load(MODELS_DIR / "q_learning_policy.pkl")
    dqn_policy = None
    if args.include_dqn and (MODELS_DIR / "dqn_policy.pt").exists():
        try:
            from dqn_policy import DQNPolicy
            dqn_policy = DQNPolicy.load(MODELS_DIR / "dqn_policy.pt")
        except Exception as exc:
            print(f"DQN unavailable in competition sensitivity: {exc}")

    episode_rows = []
    summary_rows = []
    for scenario_name in ("low", "medium", "high"):
        env = base_env.with_competition_scenario(scenario_name)
        policies = make_core_policies(
            env, args.start_zone, args.start_time,
            candidate_zones=None, include_legacy=args.include_legacy, q_policy=q_policy,
        )
        if dqn_policy is not None:
            policies.append(dqn_policy)
        for policy in policies:
            if hasattr(policy, "reset_diagnostics"):
                policy.reset_diagnostics()
            for scenario in scenarios.itertuples(index=False):
                result = env.simulate_episode(
                    policy, args.start_zone, args.start_time,
                    int(scenario.scenario_id), int(scenario.base_seed),
                    allowed_reposition_zones=None, horizon_minutes=HORIZON_MINUTES,
                    record_events=False, dynamic_candidate_count=args.candidate_zones,
                )
                row = episode_to_row(result)
                row["competition_scenario"] = scenario_name
                episode_rows.append(row)
        scenario_df = pd.DataFrame([x for x in episode_rows if x["competition_scenario"] == scenario_name])
        summary = summarize_episode_results(scenario_df)
        summary["competition_scenario"] = scenario_name
        start_metric = env.node_metric(args.start_zone, parse_clock_time(args.start_time))
        summary["start_zone_expected_wait_min"] = float(start_metric["expected_wait_min"])
        summary["start_zone_competitors_equivalent"] = float(start_metric.get("estimated_competitors_equivalent", 0.0))
        summary["learning_models_retrained_for_scenario"] = False
        summary_rows.extend(summary.to_dict("records"))

    episode_df = pd.DataFrame(episode_rows)
    episode_df.to_csv(PROCESSED_DIR / "competition_sensitivity_episode_results.csv", index=False)
    out = pd.DataFrame(summary_rows)
    out.to_csv(PROCESSED_DIR / "competition_sensitivity_results.csv", index=False)

    # Same-scenario monotonicity diagnostic. Small non-monotonic changes may be simulation noise;
    # the result table makes them visible rather than silently interpreting them as real gains.
    order = {"low": 0, "medium": 1, "high": 2}
    diagnostics = []
    for algorithm, group in out.groupby("algorithm"):
        group = group.assign(_order=group["competition_scenario"].map(order)).sort_values("_order")
        values = group["mean_net_earnings_2h"].to_numpy(float)
        diagnostics.append({
            "algorithm": algorithm,
            "low_to_medium_change": values[1] - values[0] if len(values) >= 2 else float("nan"),
            "medium_to_high_change": values[2] - values[1] if len(values) >= 3 else float("nan"),
            "nonincreasing_point_estimate": bool(all(values[i+1] <= values[i] + 1e-9 for i in range(len(values)-1))),
        })
    pd.DataFrame(diagnostics).to_csv(PROCESSED_DIR / "competition_sensitivity_diagnostics.csv", index=False)

    print(out[["competition_scenario", "algorithm", "mean_net_earnings_2h", "ci_lower", "ci_upper", "mean_waiting_minutes", "mean_empty_minutes"]].to_string(index=False))
    print(f"Saved: {PROCESSED_DIR / 'competition_sensitivity_results.csv'}")


if __name__ == "__main__":
    main()
