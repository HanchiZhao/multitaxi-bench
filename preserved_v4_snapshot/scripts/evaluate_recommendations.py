"""Evaluate all operating policies on the same two-hour scenario bank.

The script exports ordinary confidence intervals, paired comparisons against WAIT-only,
minute-accounting checks, representative full trajectories, and learning-policy coverage
statistics. Common random numbers are used so paired differences are more informative than
independent error bars.
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from config import (
    HORIZON_MINUTES, MODELS_DIR, PAIRED_COMPARISON_BASELINE,
    POLICY_CANDIDATE_ZONES, PROCESSED_DIR, RANDOM_SEED, SCENARIO_COUNT,
    ensure_directories,
)
from policies import QLearningPolicy, make_core_policies
from scenario_bank import build_scenario_bank
from two_hour_environment import DynamicTaxiEnvironment, TaxiState, WAIT_ACTION, parse_clock_time


def episode_to_row(r) -> Dict[str, object]:
    return {
        "algorithm": r.algorithm, "scenario_id": r.scenario_id,
        "start_zone": r.start_zone, "start_time": r.start_time,
        "horizon_minutes": r.horizon_minutes,
        "gross_revenue_2h": r.gross_revenue,
        "operating_cost_2h": r.operating_cost,
        "net_earnings_2h": r.net_earnings,
        "completed_trips": r.completed_trips,
        "waiting_minutes": r.waiting_minutes,
        "empty_minutes": r.empty_minutes,
        "occupied_minutes": r.occupied_minutes,
        "unfinished_occupied_minutes": r.unfinished_occupied_minutes,
        "terminal_unused_minutes": r.terminal_unused_minutes,
        "accounted_minutes": r.accounted_minutes,
        "time_accounting_error": r.time_accounting_error,
        "empty_miles": r.empty_miles,
        "occupied_miles": r.occupied_miles,
        "unfinished_trip_at_horizon": r.unfinished_trip_at_horizon,
        "end_zone": r.end_zone,
        "wait_decisions": r.wait_decisions,
        "reposition_decisions": r.reposition_decisions,
        "first_action": r.first_action,
    }


def event_to_row(algorithm: str, scenario_id: int, e) -> Dict[str, object]:
    return {
        "algorithm": algorithm, "scenario_id": scenario_id,
        "event_index": e.event_index, "event_type": e.event_type,
        "from_zone": e.from_zone, "to_zone": e.to_zone,
        "start_elapsed_min": e.start_elapsed_min, "end_elapsed_min": e.end_elapsed_min,
        "event_duration_minutes": float(e.end_elapsed_min) - float(e.start_elapsed_min),
        "start_clock": e.start_clock, "end_clock": e.end_clock,
        "revenue": e.revenue, "operating_cost": e.operating_cost,
        "distance_miles": e.distance_miles, "wait_minutes": e.wait_minutes,
        "confidence": e.confidence, "completed_within_horizon": e.completed_within_horizon,
        "route_path": e.route_path, "selected_action": e.selected_action,
    }


def summarize_episode_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, group in df.groupby("algorithm", sort=False):
        n = len(group)
        mean = float(group["net_earnings_2h"].mean())
        std = float(group["net_earnings_2h"].std(ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(max(1, n))
        total_decisions = (group["wait_decisions"] + group["reposition_decisions"]).replace(0, np.nan)
        rows.append({
            "algorithm": algorithm, "episodes": n,
            "mean_net_earnings_2h": mean, "net_earnings_std": std,
            "ci_lower": mean - 1.96 * se, "ci_upper": mean + 1.96 * se,
            "median_net_earnings_2h": float(group["net_earnings_2h"].median()),
            "worst_10_percent": float(group["net_earnings_2h"].quantile(0.10)),
            "mean_gross_revenue_2h": float(group["gross_revenue_2h"].mean()),
            "mean_operating_cost_2h": float(group["operating_cost_2h"].mean()),
            "mean_completed_trips": float(group["completed_trips"].mean()),
            "completed_trips_std": float(group["completed_trips"].std(ddof=1)) if n > 1 else 0.0,
            "mean_waiting_minutes": float(group["waiting_minutes"].mean()),
            "mean_empty_minutes": float(group["empty_minutes"].mean()),
            "mean_occupied_minutes": float(group["occupied_minutes"].mean()),
            "mean_unfinished_occupied_minutes": float(group["unfinished_occupied_minutes"].mean()),
            "mean_terminal_unused_minutes": float(group["terminal_unused_minutes"].mean()),
            "mean_accounted_minutes": float(group["accounted_minutes"].mean()),
            "max_abs_time_accounting_error": float(group["time_accounting_error"].abs().max()),
            "mean_empty_miles": float(group["empty_miles"].mean()),
            "mean_occupied_miles": float(group["occupied_miles"].mean()),
            "unfinished_trip_rate": float(group["unfinished_trip_at_horizon"].mean()),
            "mean_wait_decisions": float(group["wait_decisions"].mean()),
            "mean_reposition_decisions": float(group["reposition_decisions"].mean()),
            "reposition_decision_share": float((group["reposition_decisions"] / total_decisions).fillna(0).mean()),
            "first_action_wait_rate": float((group["first_action"].astype(str) == WAIT_ACTION).mean()),
        })
    out = pd.DataFrame(rows).sort_values("mean_net_earnings_2h", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_paired_comparisons(
    episodes_df: pd.DataFrame,
    baseline: str = PAIRED_COMPARISON_BASELINE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = episodes_df.pivot_table(
        index="scenario_id", columns="algorithm", values="net_earnings_2h", aggfunc="first"
    )
    if baseline not in pivot.columns:
        return pd.DataFrame(), pd.DataFrame()
    episode_rows = []
    summary_rows = []
    for algorithm in pivot.columns:
        if algorithm == baseline:
            continue
        paired = pivot[[algorithm, baseline]].dropna().copy()
        paired["difference"] = paired[algorithm] - paired[baseline]
        for scenario_id, row in paired.iterrows():
            episode_rows.append({
                "scenario_id": int(scenario_id), "algorithm": algorithm, "baseline": baseline,
                "algorithm_net_earnings": float(row[algorithm]),
                "baseline_net_earnings": float(row[baseline]),
                "paired_difference": float(row["difference"]),
            })
        diff = paired["difference"].to_numpy(float)
        n = len(diff)
        mean = float(np.mean(diff)) if n else np.nan
        std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(max(1, n))
        summary_rows.append({
            "algorithm": algorithm, "baseline": baseline, "paired_scenarios": n,
            "mean_paired_gain": mean, "paired_gain_std": std,
            "paired_ci_lower": mean - 1.96 * se,
            "paired_ci_upper": mean + 1.96 * se,
            "median_paired_gain": float(np.median(diff)) if n else np.nan,
            "win_rate": float(np.mean(diff > 1e-9)) if n else np.nan,
            "tie_rate": float(np.mean(np.abs(diff) <= 1e-9)) if n else np.nan,
            "loss_rate": float(np.mean(diff < -1e-9)) if n else np.nan,
        })
    return pd.DataFrame(episode_rows), pd.DataFrame(summary_rows)


def policy_similarity_diagnostics(env, policies, start_time: str, candidate_count: int) -> None:
    start_abs = parse_clock_time(start_time)
    elapsed_grid = list(range(0, HORIZON_MINUTES, 15))
    rows = []
    for elapsed in elapsed_grid:
        remaining = HORIZON_MINUTES - elapsed
        absolute = start_abs + elapsed
        for zone in env.zone_ids:
            allowed = env.dynamic_candidate_zones(zone, absolute, n=candidate_count)
            state = TaxiState(zone, float(elapsed), float(remaining))
            record = {"zone_id": zone, "elapsed_minutes": elapsed, "clock_minutes": absolute % 1440}
            for policy in policies:
                try:
                    action = policy.select_action(env, state, start_abs, allowed)
                except Exception:
                    action = "ERROR"
                record[policy.name] = str(action)
            rows.append(record)
    long = pd.DataFrame(rows)
    long.to_csv(PROCESSED_DIR / "policy_action_samples.csv", index=False)
    names = [policy.name for policy in policies]
    agreement = pd.DataFrame(index=names, columns=names, dtype=float)
    type_agreement = pd.DataFrame(index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            left_values, right_values = long[left].astype(str), long[right].astype(str)
            agreement.loc[left, right] = float((left_values == right_values).mean())
            left_type = np.where(left_values == WAIT_ACTION, "WAIT", "REPOSITION")
            right_type = np.where(right_values == WAIT_ACTION, "WAIT", "REPOSITION")
            type_agreement.loc[left, right] = float((left_type == right_type).mean())
    agreement.index.name = "algorithm"
    type_agreement.index.name = "algorithm"
    agreement.to_csv(PROCESSED_DIR / "policy_action_agreement_matrix.csv")
    type_agreement.to_csv(PROCESSED_DIR / "policy_action_type_agreement_matrix.csv")
    wait_rows = []
    for name in names:
        wait_rows.append({
            "algorithm": name, "state_count": len(long),
            "wait_action_rate": float((long[name].astype(str) == WAIT_ACTION).mean()),
            "unique_actions": int(long[name].astype(str).nunique()),
        })
    pd.DataFrame(wait_rows).to_csv(PROCESSED_DIR / "policy_action_summary.csv", index=False)


def choose_representative(group: pd.DataFrame) -> pd.Series:
    """Choose a typical episode jointly by earnings, trip count and time accounting."""
    mean_earnings = float(group["net_earnings_2h"].mean())
    std_earnings = max(1.0, float(group["net_earnings_2h"].std(ddof=1) or 0.0))
    mean_trips = float(group["completed_trips"].mean())
    std_trips = max(1.0, float(group["completed_trips"].std(ddof=1) or 0.0))
    score = (
        (group["net_earnings_2h"] - mean_earnings).abs() / std_earnings
        + (group["completed_trips"] - mean_trips).abs() / std_trips
        + 5.0 * group["time_accounting_error"].abs()
    )
    return group.loc[score.idxmin()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-hour operating policies")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--episodes", type=int, default=SCENARIO_COUNT)
    parser.add_argument("--candidate-zones", type=int, default=POLICY_CANDIDATE_ZONES)
    parser.add_argument("--include-q-learning", action="store_true")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--include-dqn", action="store_true")
    parser.add_argument("--competition-scenario", choices=["low", "medium", "high"], default=None)
    args = parser.parse_args()
    ensure_directories()

    env = DynamicTaxiEnvironment.load()
    if args.competition_scenario:
        env = env.with_competition_scenario(args.competition_scenario)
    if args.start_zone not in env.zone_to_index:
        raise ValueError(f"Unknown start zone {args.start_zone}")

    details = env.candidate_zone_details(args.start_zone, args.start_time, n=args.candidate_zones)
    if not details.empty:
        details[details["selected"]].to_csv(PROCESSED_DIR / "policy_candidate_zones.csv", index=False)

    scenario_path = PROCESSED_DIR / "scenario_bank.csv"
    if scenario_path.exists():
        scenarios = pd.read_csv(scenario_path).head(args.episodes)
    else:
        scenarios = build_scenario_bank(args.episodes, RANDOM_SEED)
    if len(scenarios) < args.episodes:
        scenarios = build_scenario_bank(args.episodes, RANDOM_SEED)
    scenarios.to_csv(scenario_path, index=False)

    q_policy = None
    if args.include_q_learning:
        q_path = MODELS_DIR / "q_learning_policy.pkl"
        if q_path.exists():
            q_policy = QLearningPolicy.load(q_path)
        else:
            print("Q-learning model missing; skipping. Run train_policies.py first.")
    policies = make_core_policies(
        env, args.start_zone, args.start_time, candidate_zones=None,
        include_legacy=args.include_legacy, q_policy=q_policy,
    )
    if args.include_dqn:
        dqn_path = MODELS_DIR / "dqn_policy.pt"
        if dqn_path.exists():
            try:
                from dqn_policy import DQNPolicy
                policies.append(DQNPolicy.load(dqn_path))
            except Exception as exc:
                print(f"Could not load DQN: {exc}")
        else:
            print("DQN model missing; skipping. Run train_dqn.py first.")

    for policy in policies:
        if hasattr(policy, "reset_diagnostics"):
            policy.reset_diagnostics()

    all_rows = []
    for policy in policies:
        print(f"Evaluating {policy.name} on {len(scenarios)} common scenarios...")
        for scenario in scenarios.itertuples(index=False):
            result = env.simulate_episode(
                policy, args.start_zone, args.start_time,
                int(scenario.scenario_id), int(scenario.base_seed),
                allowed_reposition_zones=None, horizon_minutes=HORIZON_MINUTES,
                record_events=False, dynamic_candidate_count=args.candidate_zones,
            )
            all_rows.append(episode_to_row(result))

    episodes_df = pd.DataFrame(all_rows)
    summary_df = summarize_episode_results(episodes_df)
    paired_episode, paired_summary = build_paired_comparisons(episodes_df)
    if not paired_summary.empty:
        summary_df = summary_df.merge(
            paired_summary.rename(columns={"algorithm": "algorithm"}),
            on="algorithm", how="left",
        )
    episodes_df.to_csv(PROCESSED_DIR / "policy_episode_results.csv", index=False)
    summary_df.to_csv(PROCESSED_DIR / "algorithm_recommendation_comparison.csv", index=False)
    paired_episode.to_csv(PROCESSED_DIR / "paired_policy_episode_differences.csv", index=False)
    paired_summary.to_csv(PROCESSED_DIR / "paired_policy_comparisons.csv", index=False)

    diagnostics = []
    for policy in policies:
        if hasattr(policy, "diagnostics"):
            diagnostics.append(policy.diagnostics())
    pd.DataFrame(diagnostics).to_csv(PROCESSED_DIR / "learning_policy_diagnostics.csv", index=False)

    trajectory_rows = []
    representative_rows = []
    seed_map = dict(zip(scenarios["scenario_id"].astype(int), scenarios["base_seed"].astype(int)))
    for policy in policies:
        group = episodes_df[episodes_df["algorithm"] == policy.name]
        chosen = choose_representative(group)
        scenario_id = int(chosen["scenario_id"])
        detailed = env.simulate_episode(
            policy, args.start_zone, args.start_time, scenario_id, seed_map[scenario_id],
            None, HORIZON_MINUTES, True, args.candidate_zones,
        )
        representative_rows.append({
            "algorithm": policy.name, "scenario_id": scenario_id,
            "selection_method": "jointly_closest_to_mean_earnings_and_mean_trip_count",
            "net_earnings_2h": detailed.net_earnings,
            "completed_trips": detailed.completed_trips,
            "first_action": detailed.first_action,
            "wait_decisions": detailed.wait_decisions,
            "reposition_decisions": detailed.reposition_decisions,
            "waiting_minutes": detailed.waiting_minutes,
            "empty_minutes": detailed.empty_minutes,
            "occupied_minutes": detailed.occupied_minutes,
            "unfinished_occupied_minutes": detailed.unfinished_occupied_minutes,
            "terminal_unused_minutes": detailed.terminal_unused_minutes,
            "accounted_minutes": detailed.accounted_minutes,
            "time_accounting_error": detailed.time_accounting_error,
        })
        for event in detailed.events:
            trajectory_rows.append(event_to_row(policy.name, scenario_id, event))
    trajectories = pd.DataFrame(trajectory_rows)
    trajectories.to_csv(PROCESSED_DIR / "policy_trajectories.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(PROCESSED_DIR / "representative_trajectories.csv", index=False)
    if not trajectories.empty:
        event_summary = (
            trajectories.groupby(["algorithm", "event_type"], as_index=False)
            .agg(event_count=("event_index", "size"), total_minutes=("event_duration_minutes", "sum"),
                 total_revenue=("revenue", "sum"), total_distance_miles=("distance_miles", "sum"))
        )
        event_summary.to_csv(PROCESSED_DIR / "trajectory_event_summary.csv", index=False)

    policy_similarity_diagnostics(env, policies, args.start_time, args.candidate_zones)
    print("\nPolicy evaluation complete.")
    display_cols = ["rank", "algorithm", "mean_net_earnings_2h", "ci_lower", "ci_upper", "reposition_decision_share"]
    print(summary_df[display_cols].to_string(index=False))
    if not paired_summary.empty:
        print("\nPaired gains versus WAIT-only:")
        print(paired_summary[["algorithm", "mean_paired_gain", "paired_ci_lower", "paired_ci_upper", "win_rate"]].to_string(index=False))
    for name in [
        "policy_candidate_zones.csv", "scenario_bank.csv", "policy_episode_results.csv",
        "algorithm_recommendation_comparison.csv", "paired_policy_episode_differences.csv",
        "paired_policy_comparisons.csv", "learning_policy_diagnostics.csv",
        "policy_trajectories.csv", "representative_trajectories.csv", "trajectory_event_summary.csv",
        "policy_action_samples.csv", "policy_action_agreement_matrix.csv",
        "policy_action_type_agreement_matrix.csv", "policy_action_summary.csv",
    ]:
        print(f"  {PROCESSED_DIR / name}")


if __name__ == "__main__":
    main()
