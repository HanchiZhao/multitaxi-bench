"""Dynamic Zone Shapley for two-hour expected net earnings.

A player is a taxi zone that may be used as an active empty-reposition destination.
Passenger destinations remain unrestricted. The coalition value is the highest expected
120-minute net earnings obtainable by finite-horizon dynamic programming when only zones
in the coalition are available for active repositioning.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    HORIZON_MINUTES,
    PROCESSED_DIR,
    RANDOM_SEED,
    SHAPLEY_CANDIDATE_ZONES,
    SHAPLEY_EXACT_MAX_PLAYERS,
    SHAPLEY_PERMUTATIONS,
    SHAPLEY_POOL_MULTIPLIER,
    ensure_directories,
)
from scenario_bank import build_scenario_bank
from two_hour_environment import (
    DynamicProgramPolicy, DynamicTaxiEnvironment, FiniteHorizonModel,
    WAIT_ACTION, parse_clock_time,
)


def coalition_mask(zones: Sequence[int], coalition: Iterable[int]) -> int:
    selected = set(int(z) for z in coalition)
    mask = 0
    for i, zone in enumerate(zones):
        if zone in selected:
            mask |= 1 << i
    return mask


def zones_from_mask(zones: Sequence[int], mask: int) -> Tuple[int, ...]:
    return tuple(int(zone) for i, zone in enumerate(zones) if mask & (1 << i))


class CoalitionValueCache:
    def __init__(
        self,
        model: FiniteHorizonModel,
        start_zone: int,
        player_zones: Sequence[int],
        base_allowed_zones: Sequence[int] = (),
    ) -> None:
        self.model = model
        self.start_zone = int(start_zone)
        self.player_zones = tuple(int(z) for z in player_zones)
        self.base_allowed_zones = tuple(int(z) for z in base_allowed_zones)
        self.values: Dict[int, float] = {}

    def value(self, mask: int) -> float:
        mask = int(mask)
        if mask not in self.values:
            coalition = zones_from_mask(self.player_zones, mask)
            self.values[mask] = self.model.start_value(
                self.start_zone,
                coalition,
                base_allowed_zone_ids=self.base_allowed_zones,
            )
        return self.values[mask]


def select_candidate_players(
    env: DynamicTaxiEnvironment,
    start_zone: int,
    start_time: str,
    n_players: int,
) -> Tuple[List[int], pd.DataFrame]:
    """Select players from a multi-state, multi-criterion dynamic pool.

    The previous version selected players almost entirely by singleton value from the initial
    state. That could omit zones useful only after a passenger moved the driver elsewhere, or
    zones valuable through interaction. This version scans reachable states across the full
    horizon and combines coverage, singleton gain and one-step pair synergy.
    """
    start_abs = parse_clock_time(start_time)
    pool_target = max(int(n_players) * SHAPLEY_POOL_MULTIPLIER, int(n_players) + 8)
    coverage = defaultdict(int)
    discovery = defaultdict(set)
    visited = {int(start_zone)}
    frontier = {int(start_zone)}
    elapsed_grid = [0, 30, 60, 90]

    for depth in range(3):
        next_frontier = set()
        for zone in sorted(frontier):
            for elapsed in elapsed_grid:
                absolute = start_abs + elapsed
                candidates = env.dynamic_candidate_zones(zone, absolute, n=max(8, n_players))
                for z in candidates:
                    coverage[int(z)] += 1
                    discovery[int(z)].add(f"dynamic_depth_{depth}")
                    next_frontier.add(int(z))
                # Passenger destinations expose states the driver can reach without actively
                # choosing them; their subsequent local candidates matter for the 2h game.
                for option in env.od_options(zone, absolute)[:4]:
                    d = int(option["destination"])
                    coverage[d] += 1
                    discovery[d].add("passenger_reachable")
                    next_frontier.add(d)
        visited |= next_frontier
        frontier = set(sorted(next_frontier, key=lambda z: coverage[z], reverse=True)[:max(12, n_players*2)])

    pool = [z for z,_ in sorted(coverage.items(), key=lambda kv:(kv[1],-kv[0]), reverse=True) if z != int(start_zone)]
    if len(pool) < pool_target:
        for z in env.dynamic_candidate_zones(start_zone, start_abs, n=pool_target):
            if z not in pool and z != int(start_zone): pool.append(z)
    pool = pool[:pool_target]
    if not pool:
        raise RuntimeError("No feasible dynamic Shapley player pool was found")

    pool_model = FiniteHorizonModel(env, start_time, HORIZON_MINUTES, pool)
    baseline = pool_model.start_value(start_zone, [])
    rows = []
    singleton_values = {}
    for zone in pool:
        singleton = pool_model.start_value(start_zone, [zone])
        singleton_values[zone] = singleton
        route = env.route(start_zone, zone, absolute_minutes=start_abs)
        rows.append({
            "zone_id": zone,
            "zone_name": env.zone_names.get(zone,str(zone)),
            "borough": env.zone_boroughs.get(zone,""),
            "state_coverage_count": int(coverage.get(zone,0)),
            "discovery_sources": "|".join(sorted(discovery.get(zone,set()))),
            "singleton_value": singleton,
            "singleton_gain_over_wait_only": singleton-baseline,
            "reposition_time_from_start": route["duration_min"],
            "reposition_distance_from_start": route["distance_miles"],
        })
    ranking = pd.DataFrame(rows)
    best_anchor = max(pool, key=lambda z: singleton_values[z])
    anchor_value = singleton_values[best_anchor]
    synergy = {}
    for zone in pool:
        if zone == best_anchor:
            synergy[zone] = 0.0
        else:
            pair = pool_model.start_value(start_zone,[best_anchor,zone])
            synergy[zone] = pair - anchor_value - singleton_values[zone] + baseline
    ranking["pair_synergy_with_best_anchor"] = ranking["zone_id"].map(synergy).fillna(0.0)

    # Allocate roughly one third of players to each evidence channel.
    chosen=[]; reasons=defaultdict(list)
    quota=max(1,int(math.ceil(n_players/3)))
    for metric,reason in [
        ("singleton_gain_over_wait_only","singleton_gain"),
        ("state_coverage_count","multi_state_coverage"),
        ("pair_synergy_with_best_anchor","pair_synergy"),
    ]:
        for z in ranking.nlargest(quota,metric)["zone_id"].astype(int):
            if z not in chosen and len(chosen)<n_players:
                chosen.append(z)
            reasons[z].append(reason)
    # Fill remaining slots by a rank aggregate.
    ranking["rank_singleton"] = ranking["singleton_gain_over_wait_only"].rank(method="min",ascending=False)
    ranking["rank_coverage"] = ranking["state_coverage_count"].rank(method="min",ascending=False)
    ranking["rank_synergy"] = ranking["pair_synergy_with_best_anchor"].rank(method="min",ascending=False)
    ranking["selection_rank_sum"] = ranking[["rank_singleton","rank_coverage","rank_synergy"]].sum(axis=1)
    for z in ranking.sort_values("selection_rank_sum")["zone_id"].astype(int):
        if z not in chosen:
            chosen.append(z); reasons[z].append("rank_aggregate")
        if len(chosen)>=int(n_players): break
    ranking["selected_as_shapley_player"] = ranking["zone_id"].isin(chosen)
    ranking["selection_reason"] = ranking["zone_id"].map(lambda z:"|".join(reasons.get(int(z),[])))
    order={z:i+1 for i,z in enumerate(chosen)}
    ranking["selected_player_order"] = ranking["zone_id"].map(order)
    ranking = ranking.sort_values(["selected_as_shapley_player","selected_player_order","selection_rank_sum"],ascending=[False,True,True])
    return chosen, ranking.reset_index(drop=True)

def exact_shapley(cache: CoalitionValueCache) -> Tuple[np.ndarray, np.ndarray]:
    n = len(cache.player_zones)
    total_masks = 1 << n
    for mask in range(total_masks):
        cache.value(mask)
    factorial = math.factorial
    denom = factorial(n)
    phi = np.zeros(n, dtype=float)
    for i in range(n):
        bit = 1 << i
        for mask in range(total_masks):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = factorial(size) * factorial(n - size - 1) / denom
            phi[i] += weight * (cache.value(mask | bit) - cache.value(mask))
    return phi, np.zeros(n, dtype=float)


def monte_carlo_shapley(
    cache: CoalitionValueCache,
    permutations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(cache.player_zones)
    rng = np.random.default_rng(int(seed))
    samples: List[np.ndarray] = []
    requested = max(1, int(permutations))
    for _ in range((requested + 1) // 2):
        perm = rng.permutation(n)
        for order in (perm, perm[::-1]):
            contrib = np.zeros(n, dtype=float)
            mask = 0
            before = cache.value(mask)
            for idx in order:
                new_mask = mask | (1 << int(idx))
                after = cache.value(new_mask)
                contrib[int(idx)] = after - before
                mask = new_mask
                before = after
            samples.append(contrib)
            if len(samples) >= requested:
                break
        if len(samples) >= requested:
            break
    matrix = np.vstack(samples)
    mean = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix)) if len(matrix) > 1 else np.zeros(n)
    return mean, se, len(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Dynamic Zone Shapley")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--players", type=int, default=SHAPLEY_CANDIDATE_ZONES)
    parser.add_argument("--method", choices=["auto", "exact", "mc"], default="auto")
    parser.add_argument("--permutations", type=int, default=SHAPLEY_PERMUTATIONS)
    parser.add_argument("--calibration-episodes", type=int, default=150,
                        help="Common scenarios used to compare DP model value with simulation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--competition-scenario", choices=["low", "medium", "high"], default=None)
    parser.add_argument(
        "--base-zone",
        type=int,
        action="append",
        default=[],
        help="Optional baseline reposition zone always available; repeat flag for multiple zones",
    )
    args = parser.parse_args()
    ensure_directories()
    env = DynamicTaxiEnvironment.load()
    if args.competition_scenario:
        env = env.with_competition_scenario(args.competition_scenario)
    if args.start_zone not in env.zone_to_index:
        raise ValueError(f"Unknown start zone {args.start_zone}")

    players, ranking = select_candidate_players(
        env, args.start_zone, args.start_time, args.players
    )
    ranking.to_csv(PROCESSED_DIR / "shapley_candidate_zones.csv", index=False)
    model = FiniteHorizonModel(env, args.start_time, HORIZON_MINUTES, players)
    cache = CoalitionValueCache(
        model,
        start_zone=args.start_zone,
        player_zones=players,
        base_allowed_zones=args.base_zone,
    )
    method = args.method
    if method == "auto":
        method = "exact" if len(players) <= SHAPLEY_EXACT_MAX_PLAYERS else "mc"
    if method == "exact":
        print(f"Computing exact Shapley for {len(players)} zones ({1 << len(players)} coalitions)...")
        phi, se = exact_shapley(cache)
        n_permutations = math.factorial(len(players))
    else:
        print(f"Computing Monte Carlo Shapley with {args.permutations} antithetic permutations...")
        phi, se, n_permutations = monte_carlo_shapley(
            cache, args.permutations, args.seed
        )

    empty_value = cache.value(0)
    full_mask = (1 << len(players)) - 1
    full_value = cache.value(full_mask)
    results = pd.DataFrame(
        {
            "zone_id": players,
            "zone_name": [env.zone_names.get(z, str(z)) for z in players],
            "borough": [env.zone_boroughs.get(z, "") for z in players],
            "shapley_value": phi,
            "shapley_standard_error": se,
            "ci_lower": phi - 1.96 * se,
            "ci_upper": phi + 1.96 * se,
        }
    ).sort_values("shapley_value", ascending=False).reset_index(drop=True)
    results.insert(0, "shapley_rank", np.arange(1, len(results) + 1))
    results["start_zone"] = int(args.start_zone)
    results["start_time"] = str(args.start_time)
    results["value_unit"] = "USD_per_2h"
    results["competition_scenario"] = env.competition_scenario
    results["method"] = method
    results["num_permutations"] = int(n_permutations)
    results["candidate_zone_count"] = len(players)
    results["baseline_value"] = empty_value
    results["full_coalition_value"] = full_value
    selected_diagnostics = ranking[ranking["selected_as_shapley_player"]].copy()
    merge_columns = [
        "zone_id", "state_coverage_count", "discovery_sources", "singleton_value",
        "singleton_gain_over_wait_only", "pair_synergy_with_best_anchor",
        "reposition_time_from_start", "reposition_distance_from_start",
        "selection_reason", "selected_player_order",
    ]
    results = results.merge(selected_diagnostics[merge_columns], on="zone_id", how="left")
    results.to_csv(PROCESSED_DIR / "node_shapley_values.csv", index=False)

    # Zone diagnostics make high contributions auditable, especially for sparse zones.
    start_absolute = parse_clock_time(args.start_time)
    diagnostic_rows = []
    for row in results.itertuples(index=False):
        zone = int(row.zone_id)
        metric = env.node_metric(zone, start_absolute)
        origin_rows = env.od_metrics[env.od_metrics["origin"].astype(int) == zone]
        diagnostic_rows.append({
            "zone_id": zone, "zone_name": env.zone_names.get(zone, str(zone)),
            "shapley_value": float(row.shapley_value),
            "start_time_expected_wait_min": float(metric["expected_wait_min"]),
            "start_time_pickup_rate_index": float(metric["pickup_rate_index"]),
            "start_time_competition_ratio": float(metric["competition_ratio"]),
            "start_time_estimated_competitors_equivalent": float(metric.get("estimated_competitors_equivalent", 0.0)),
            "start_time_node_confidence": float(metric["confidence"]),
            "origin_dynamic_od_rows": int(len(origin_rows)),
            "origin_observed_trip_count_total": float(origin_rows["observed_trip_count"].sum()) if not origin_rows.empty else 0.0,
            "origin_mean_od_confidence": float(origin_rows["confidence"].mean()) if not origin_rows.empty else 0.0,
            "singleton_gain_over_wait_only": float(getattr(row, "singleton_gain_over_wait_only", 0.0)),
            "pair_synergy_with_best_anchor": float(getattr(row, "pair_synergy_with_best_anchor", 0.0)),
            "state_coverage_count": int(getattr(row, "state_coverage_count", 0)),
            "selection_reason": str(getattr(row, "selection_reason", "")),
        })
    pd.DataFrame(diagnostic_rows).to_csv(PROCESSED_DIR / "shapley_zone_diagnostics.csv", index=False)

    coalition_rows = []
    for mask, value in sorted(cache.values.items()):
        coalition = zones_from_mask(players, mask)
        coalition_rows.append(
            {
                "coalition_mask": mask,
                "coalition_size": len(coalition),
                "coalition_zone_ids": "|".join(str(z) for z in coalition),
                "coalition_value": value,
                "base_allowed_zone_ids": "|".join(str(z) for z in args.base_zone),
            }
        )
    pd.DataFrame(coalition_rows).to_csv(
        PROCESSED_DIR / "shapley_coalition_values.csv", index=False
    )

    efficiency_gap = float(phi.sum() - (full_value - empty_value))
    summary = pd.DataFrame(
        [
            {
                "start_zone": args.start_zone,
                "start_zone_name": env.zone_names.get(args.start_zone, str(args.start_zone)),
                "start_time": args.start_time,
                "horizon_minutes": HORIZON_MINUTES,
                "method": method,
                "candidate_zone_count": len(players),
                "coalitions_evaluated": len(cache.values),
                "num_permutations": n_permutations,
                "baseline_value_wait_only": empty_value,
                "full_coalition_value": full_value,
                "repositioning_value_gain": full_value - empty_value,
                "sum_shapley": float(phi.sum()),
                "efficiency_gap": efficiency_gap,
                "value_unit": "USD_per_2h",
                "competition_scenario": env.competition_scenario,
                "base_allowed_zone_ids": "|".join(str(z) for z in args.base_zone),
            }
        ]
    )
    summary.to_csv(PROCESSED_DIR / "shapley_run_summary.csv", index=False)

    # Model-versus-simulation calibration on the exact same restricted action set.
    class _WaitOnly:
        name = "wait_only"
        def select_action(self, env, state, start_absolute_minutes, allowed_zones):
            return WAIT_ACTION

    calibration_count = max(1, int(args.calibration_episodes))
    scenario_path = PROCESSED_DIR / "scenario_bank.csv"
    if scenario_path.exists():
        scenarios = pd.read_csv(scenario_path).head(calibration_count)
    else:
        scenarios = build_scenario_bank(calibration_count, args.seed)
    if len(scenarios) < calibration_count:
        scenarios = build_scenario_bank(calibration_count, args.seed)
    full_policy = DynamicProgramPolicy(model, players, args.base_zone)
    fixed_allowed = list(dict.fromkeys([*args.base_zone, *players]))
    differences = []
    full_values = []
    baseline_values = []
    for scenario in scenarios.itertuples(index=False):
        full_result = env.simulate_episode(
            full_policy, args.start_zone, args.start_time, int(scenario.scenario_id), int(scenario.base_seed),
            allowed_reposition_zones=fixed_allowed, horizon_minutes=HORIZON_MINUTES,
            record_events=False, dynamic_candidate_count=len(fixed_allowed),
        )
        baseline_result = env.simulate_episode(
            _WaitOnly(), args.start_zone, args.start_time, int(scenario.scenario_id), int(scenario.base_seed),
            allowed_reposition_zones=[], horizon_minutes=HORIZON_MINUTES,
            record_events=False, dynamic_candidate_count=0,
        )
        full_values.append(full_result.net_earnings)
        baseline_values.append(baseline_result.net_earnings)
        differences.append(full_result.net_earnings - baseline_result.net_earnings)
    diff = np.asarray(differences, dtype=float)
    diff_std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    diff_se = diff_std / math.sqrt(max(1, len(diff)))
    calibration = pd.DataFrame([{
        "start_zone": args.start_zone, "start_time": args.start_time,
        "calibration_episodes": len(diff),
        "dp_expected_baseline_value": empty_value,
        "dp_expected_full_coalition_value": full_value,
        "dp_expected_repositioning_gain": full_value - empty_value,
        "simulation_mean_baseline_value": float(np.mean(baseline_values)),
        "simulation_mean_full_coalition_value": float(np.mean(full_values)),
        "simulation_mean_paired_gain": float(np.mean(diff)),
        "simulation_paired_ci_lower": float(np.mean(diff) - 1.96 * diff_se),
        "simulation_paired_ci_upper": float(np.mean(diff) + 1.96 * diff_se),
        "simulation_win_rate": float(np.mean(diff > 1e-9)),
        "action_scope": "restricted_to_selected_shapley_players_plus_base_zones",
    }])
    calibration.to_csv(PROCESSED_DIR / "shapley_value_calibration.csv", index=False)

    print("\nDynamic Zone Shapley complete.")
    print(results[["shapley_rank", "zone_id", "zone_name", "shapley_value", "ci_lower", "ci_upper"]].to_string(index=False))
    print(f"Efficiency gap: {efficiency_gap:.10f}")
    for name in [
        "shapley_candidate_zones.csv",
        "node_shapley_values.csv",
        "shapley_coalition_values.csv",
        "shapley_run_summary.csv",
        "shapley_zone_diagnostics.csv",
        "shapley_value_calibration.csv",
    ]:
        print(f"  {PROCESSED_DIR / name}")


if __name__ == "__main__":
    main()
