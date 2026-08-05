"""Dynamic two-hour taxi operating environment.

The driver starts empty at an arbitrary taxi zone and clock time. At each empty state the
policy may WAIT or REPOSITION. Passenger destinations are sampled from historical/EB OD
probabilities and are never chosen by the policy.

Version 4 final changes:
- arbitrary origin-destination-time estimation via estimate_od();
- latent vacant-taxi competition in waiting times;
- dynamic state-specific reposition candidates;
- time-of-day routing speeds;
- a 5-minute semi-Markov DP with interpolation rather than 15-minute rounding.
"""
from __future__ import annotations

import copy
import hashlib
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import (
    COMPETITION_SCENARIO_MULTIPLIERS,
    DECISION_TIME_STEP_MINUTES,
    DEFAULT_COMPETITION_SCENARIO,
    DYNAMIC_CANDIDATE_CATEGORY_QUOTA,
    DURATION_DISTANCE_BINS_MILES,
    DURATION_MAX_TAIL_MULTIPLIER,
    DURATION_MIN_TAIL_MULTIPLIER,
    EMPTY_COST_PER_MILE,
    FALLBACK_SPEED_MPH,
    FULL_OD_CACHE_MAX_SIZE,
    HORIZON_MINUTES,
    MAX_REPOSITION_MILES,
    MAX_REPOSITION_MINUTES,
    MAX_SPEED_MPH,
    MIN_CONFIDENCE,
    MIN_SPEED_MPH,
    OCCUPIED_COST_PER_MILE,
    OD_OBS_PRIOR_STRENGTH,
    POLICY_CANDIDATE_ZONES,
    PROCESSED_DIR,
    SPATIAL_DECAY_MILES,
    SPATIAL_NEIGHBOR_COUNT,
    TEMPORAL_DECAY_BINS,
    TEMPORAL_WINDOW_BINS,
    TIME_BIN_MINUTES,
    WAIT_REEVALUATION_MINUTES,
)

WAIT_ACTION = "WAIT"


def parse_clock_time(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value) % 1440.0
    text = str(value).strip()
    if ":" not in text:
        return float(text) % 1440.0
    hour, minute = text.split(":", 1)
    h, m = int(hour), int(minute)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid clock time: {value}")
    return float(h * 60 + m)


def format_clock_time(minutes: float) -> str:
    minutes = float(minutes) % 1440.0
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    return f"{h:02d}:{m:02d}"


@dataclass(frozen=True)
class TaxiState:
    zone_id: int
    elapsed_minutes: float
    remaining_minutes: float


@dataclass
class TrajectoryEvent:
    event_index: int
    event_type: str
    from_zone: int
    to_zone: int
    start_elapsed_min: float
    end_elapsed_min: float
    start_clock: str
    end_clock: str
    revenue: float = 0.0
    operating_cost: float = 0.0
    distance_miles: float = 0.0
    wait_minutes: float = 0.0
    confidence: float = 1.0
    completed_within_horizon: bool = True
    route_path: str = ""
    selected_action: str = ""


@dataclass
class EpisodeResult:
    algorithm: str
    scenario_id: int
    start_zone: int
    start_time: str
    horizon_minutes: int
    gross_revenue: float
    operating_cost: float
    net_earnings: float
    completed_trips: int
    waiting_minutes: float
    empty_minutes: float
    occupied_minutes: float
    empty_miles: float
    occupied_miles: float
    unfinished_trip_at_horizon: int
    end_zone: int
    unfinished_occupied_minutes: float = 0.0
    terminal_unused_minutes: float = 0.0
    accounted_minutes: float = 0.0
    time_accounting_error: float = 0.0
    wait_decisions: int = 0
    reposition_decisions: int = 0
    first_action: str = WAIT_ACTION
    events: List[TrajectoryEvent] = field(default_factory=list)


class ScenarioRandomness:
    """Deterministic common random numbers keyed by scenario, event and draw name."""

    def __init__(self, scenario_id: int, base_seed: int):
        self.scenario_id = int(scenario_id)
        self.base_seed = int(base_seed)

    def _seed(self, event_index: int, draw_name: str) -> int:
        raw = f"{self.base_seed}|{self.scenario_id}|{event_index}|{draw_name}".encode()
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    def uniform(self, event_index: int, draw_name: str) -> float:
        return float(np.random.default_rng(self._seed(event_index, draw_name)).random())

    def normal(self, event_index: int, draw_name: str) -> float:
        return float(np.random.default_rng(self._seed(event_index, draw_name)).standard_normal())


@dataclass
class DynamicTaxiEnvironment:
    node_metrics: pd.DataFrame
    od_metrics: pd.DataFrame
    reposition_edges: pd.DataFrame
    zone_centroids: pd.DataFrame
    metadata: Dict[str, object]
    global_od_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    observed_od_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.zone_ids = sorted(int(x) for x in self.zone_centroids["zone_id"].unique())
        self.zone_to_index = {z: i for i, z in enumerate(self.zone_ids)}
        self.index_to_zone = {i: z for z, i in self.zone_to_index.items()}
        self.zone_names = {int(r.zone_id): str(r.zone_name) for r in self.zone_centroids.itertuples(index=False)}
        self.zone_boroughs = {int(r.zone_id): str(r.borough) for r in self.zone_centroids.itertuples(index=False)}
        self.centroids_xy = {int(r.zone_id): (float(r.centroid_x), float(r.centroid_y)) for r in self.zone_centroids.itertuples(index=False)}
        self.centroids_lonlat = {int(r.zone_id): (float(r.centroid_lon), float(r.centroid_lat)) for r in self.zone_centroids.itertuples(index=False)}
        self._nearest_zones: Dict[int, List[Tuple[int, float]]] = {}
        for zone in self.zone_ids:
            x, y = self.centroids_xy[zone]
            candidates = []
            for other in self.zone_ids:
                if other == zone:
                    continue
                ox, oy = self.centroids_xy[other]
                candidates.append((other, math.hypot(x-ox, y-oy)/5280.0))
            candidates.sort(key=lambda item: item[1])
            self._nearest_zones[zone] = candidates[:SPATIAL_NEIGHBOR_COUNT]
            if not hasattr(self, "_candidate_nearest_zones"):
                self._candidate_nearest_zones = {}
            self._candidate_nearest_zones[zone] = candidates[:max(24, POLICY_CANDIDATE_ZONES * 2)]

        self.competition_scenario = str(self.metadata.get("competition_scenario", DEFAULT_COMPETITION_SCENARIO))
        if self.competition_scenario not in COMPETITION_SCENARIO_MULTIPLIERS:
            self.competition_scenario = DEFAULT_COMPETITION_SCENARIO

        self._node_map: Dict[Tuple[int, int], Dict[str, float]] = {}
        for r in self.node_metrics.itertuples(index=False):
            d = {
                "pickup_rate_index": float(r.posterior_pickups_per_bin_day),
                "expected_trip_revenue": float(r.expected_trip_revenue),
                "expected_trip_duration_min": float(r.expected_trip_duration_min),
                "expected_trip_distance_miles": float(r.expected_trip_distance_miles),
                "confidence": float(r.node_confidence),
                "latent_vacant_supply_index": float(getattr(r, "latent_vacant_supply_index", 0.0)),
                "competition_ratio": float(getattr(r, "competition_ratio", 1.0)),
                "raw_pickups_per_bin_day": float(getattr(r, "raw_pickups_per_bin_day", 0.0)),
                "raw_dropoffs_per_bin_day": float(getattr(r, "raw_dropoffs_per_bin_day", 0.0)),
                "estimated_competitors_equivalent": float(getattr(r, "estimated_competitors_equivalent", 0.0)),
            }
            for scenario in COMPETITION_SCENARIO_MULTIPLIERS:
                d[f"expected_wait_min_{scenario}"] = float(
                    getattr(r, f"expected_wait_min_{scenario}", getattr(r, "expected_wait_min", 12.0))
                )
                d[f"estimated_competitors_equivalent_{scenario}"] = float(
                    getattr(r, f"estimated_competitors_equivalent_{scenario}", getattr(r, "estimated_competitors_equivalent", 0.0))
                )
            self._node_map[(int(r.time_bin), int(r.zone_id))] = d

        # Precompute a compact, diverse citywide leader pool per 15-minute bin. Candidate
        # generation then evaluates routes only for this pool plus local neighbours and likely
        # passenger destinations, rather than scanning every zone at every decision.
        self._candidate_leaders_by_bin: Dict[int, List[int]] = {}
        leader_quota = max(8, POLICY_CANDIDATE_ZONES)
        for tb in range(96):
            rows = []
            for zone in self.zone_ids:
                metric = self._node_map.get((tb, zone))
                if metric is None:
                    continue
                wait = max(1e-6, metric.get(f"expected_wait_min_{self.competition_scenario}", 12.0))
                net = metric["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE * metric["expected_trip_distance_miles"]
                rows.append((
                    zone, metric["pickup_rate_index"], net, net / max(1.0, wait + metric["expected_trip_duration_min"]),
                    1.0 / max(1e-6, metric["competition_ratio"]), metric["confidence"],
                ))
            leaders: List[int] = []
            if rows:
                for index in range(1, 6):
                    for row in sorted(rows, key=lambda x: x[index], reverse=True)[:leader_quota]:
                        if row[0] not in leaders:
                            leaders.append(int(row[0]))
            self._candidate_leaders_by_bin[tb] = leaders

        self._od_map: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
        self._od_pair_map: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        for (tb, origin), group in self.od_metrics.groupby(["time_bin", "origin"], sort=False):
            rows: List[Dict[str, float]] = []
            for r in group.itertuples(index=False):
                rec = {
                    "destination": int(r.destination),
                    "probability": float(r.destination_probability),
                    "revenue": float(r.avg_driver_revenue),
                    "duration_min": float(r.avg_duration_min),
                    "distance_miles": float(r.avg_distance_miles),
                    "revenue_std": float(getattr(r, "revenue_std", 0.0) or 0.0),
                    "duration_std": float(getattr(r, "duration_std", 0.0) or 0.0),
                    "distance_std": float(getattr(r, "distance_std", 0.0) or 0.0),
                    "confidence": float(r.confidence),
                    "estimate_source": str(getattr(r, "imputation_layer", "dynamic_eb")),
                    "observed_trip_count": float(getattr(r, "observed_trip_count", 0.0)),
                }
                rows.append(rec)
                self._od_pair_map[(int(tb), int(origin), int(r.destination))] = rec
            total = sum(max(0.0, x["probability"]) for x in rows)
            if total > 0:
                for x in rows:
                    x["probability"] = max(0.0, x["probability"]) / total
            self._od_map[(int(tb), int(origin))] = rows

        self._global_od_map: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._global_origin_lists: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        if self.global_od_metrics is not None and not self.global_od_metrics.empty:
            for r in self.global_od_metrics.itertuples(index=False):
                rec = {
                    "trip_count": float(r.trip_count),
                    "revenue": float(r.avg_driver_revenue),
                    "duration_min": float(r.avg_duration_min),
                    "distance_miles": float(r.avg_distance_miles),
                    "revenue_std": float(getattr(r, "revenue_std", 0.0)),
                    "duration_std": float(getattr(r, "duration_std", 0.0)),
                    "distance_std": float(getattr(r, "distance_std", 0.0)),
                }
                self._global_od_map[(int(r.origin), int(r.destination))] = rec
                self._global_origin_lists[int(r.origin)].append((int(r.destination), float(r.trip_count)))
            for o in self._global_origin_lists:
                self._global_origin_lists[o].sort(key=lambda x: x[1], reverse=True)

        self._observed_od_map: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self._observed_pair_rows: Dict[Tuple[int, int], List[Dict[str, float]]] = defaultdict(list)
        if self.observed_od_metrics is not None and not self.observed_od_metrics.empty:
            for r in self.observed_od_metrics.itertuples(index=False):
                rec = {
                    "time_bin": int(r.time_bin),
                    "trip_count": float(r.trip_count),
                    "revenue": float(r.observed_avg_driver_revenue),
                    "duration_min": float(r.observed_avg_duration_min),
                    "distance_miles": float(r.observed_avg_distance_miles),
                    "revenue_std": float(getattr(r, "observed_revenue_std", 0.0) or 0.0),
                    "duration_std": float(getattr(r, "observed_duration_std", 0.0) or 0.0),
                    "distance_std": float(getattr(r, "observed_distance_std", 0.0) or 0.0),
                }
                key = (int(r.time_bin), int(r.origin), int(r.destination))
                self._observed_od_map[key] = rec
                self._observed_pair_rows[(int(r.origin), int(r.destination))].append(rec)

        self._reposition_graph = nx.DiGraph()
        self._reposition_graph.add_nodes_from(self.zone_ids)
        for r in self.reposition_edges.itertuples(index=False):
            self._reposition_graph.add_edge(
                int(r.origin), int(r.destination),
                duration_min=float(r.duration_min),
                distance_miles=float(r.distance_miles),
                confidence=float(getattr(r, "confidence", 1.0)),
            )
        self._route_cache: Dict[Tuple[int, int, int, str], Dict[str, object]] = {}
        self._full_od_cache: Dict[Tuple[int, int, int], Dict[str, float | str]] = {}
        self._candidate_cache: Dict[Tuple[int, int, int], List[int]] = {}

        self._speed_by_bin = {int(k): float(v) for k, v in dict(self.metadata.get("time_speed_mph", {})).items()}
        self._rpm_by_bin = {int(k): float(v) for k, v in dict(self.metadata.get("time_revenue_per_mile", {})).items()}
        self._global_speed = float(self.metadata.get("global_speed_mph", FALLBACK_SPEED_MPH))
        self._global_rpm = float(np.mean(list(self._rpm_by_bin.values())) if self._rpm_by_bin else 4.0)
        self._gravity = dict(self.metadata.get("gravity_model", {}))

    @classmethod
    def load(cls, path: Path | str | None = None) -> "DynamicTaxiEnvironment":
        path = Path(path or (PROCESSED_DIR / "dynamic_environment.pkl"))
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run: python scripts\\route_environment.py")
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            if not hasattr(obj, "global_od_metrics"):
                obj.global_od_metrics = pd.DataFrame()
            if not hasattr(obj, "observed_od_metrics"):
                obj.observed_od_metrics = pd.DataFrame()
            obj._build_indexes()
            return obj
        if isinstance(obj, Mapping):
            return cls(**obj)
        raise TypeError(f"Unsupported environment pickle type: {type(obj)!r}")

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or (PROCESSED_DIR / "dynamic_environment.pkl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def clone(self) -> "DynamicTaxiEnvironment":
        return DynamicTaxiEnvironment(
            node_metrics=self.node_metrics.copy(deep=True),
            od_metrics=self.od_metrics.copy(deep=True),
            reposition_edges=self.reposition_edges.copy(deep=True),
            zone_centroids=self.zone_centroids.copy(deep=True),
            metadata=copy.deepcopy(self.metadata),
            global_od_metrics=self.global_od_metrics.copy(deep=True),
            observed_od_metrics=self.observed_od_metrics.copy(deep=True),
        )

    def with_competition_scenario(self, scenario: str) -> "DynamicTaxiEnvironment":
        if scenario not in COMPETITION_SCENARIO_MULTIPLIERS:
            raise ValueError(f"Unknown competition scenario: {scenario}")
        clone = self.clone()
        clone.metadata["competition_scenario"] = scenario
        clone._build_indexes()
        return clone

    def time_bin(self, absolute_minutes: float) -> Tuple[int, int, float]:
        raw = (float(absolute_minutes) % 1440.0) / TIME_BIN_MINUTES
        low = int(math.floor(raw)) % 96
        return low, (low + 1) % 96, float(raw - math.floor(raw))

    def _speed(self, absolute_minutes: float) -> float:
        low, high, frac = self.time_bin(absolute_minutes)
        a = self._speed_by_bin.get(low, self._global_speed)
        b = self._speed_by_bin.get(high, self._global_speed)
        return float(np.clip((1-frac)*a + frac*b, MIN_SPEED_MPH, MAX_SPEED_MPH))

    def _rpm(self, absolute_minutes: float) -> float:
        low, high, frac = self.time_bin(absolute_minutes)
        a = self._rpm_by_bin.get(low, self._global_rpm)
        b = self._rpm_by_bin.get(high, self._global_rpm)
        return max(0.1, float((1-frac)*a + frac*b))

    @staticmethod
    def _duration_distance_band(distance_miles: float) -> str:
        edges = list(DURATION_DISTANCE_BINS_MILES)
        d = max(0.0, float(distance_miles))
        for low, high in zip(edges[:-1], edges[1:]):
            if low <= d < high:
                return f"{low:g}-{high:g}"
        return f"{edges[-2]:g}-{edges[-1]:g}"

    def _duration_calibration_multiplier(self, origin: int, destination: int, tb: int, distance_miles: float) -> float:
        calibration = dict(self.metadata.get("duration_calibration", {}) or {})
        if not calibration:
            return 1.0
        band = self._duration_distance_band(distance_miles)
        pair = f"{self.zone_boroughs.get(int(origin), 'Unknown')}->{self.zone_boroughs.get(int(destination), 'Unknown')}"
        airport = "airport" if ("airport" in self.zone_names.get(int(origin), "").lower() or
                                "airport" in self.zone_names.get(int(destination), "").lower()) else "non_airport"
        def lookup(group: str, key: object) -> float:
            values = dict(calibration.get(group, {}) or {})
            return float(values.get(key, values.get(str(key), 1.0)))
        multiplier = float(calibration.get("global_ratio", 1.0))
        multiplier *= lookup("distance", band)
        multiplier *= lookup("borough_pair", pair)
        multiplier *= lookup("airport", airport)
        multiplier *= lookup("time_bin", int(tb))
        return float(np.clip(multiplier, DURATION_MIN_TAIL_MULTIPLIER, DURATION_MAX_TAIL_MULTIPLIER))

    def calibrated_structural_duration(self, origin: int, destination: int, absolute_minutes: float, distance_miles: float) -> float:
        low, _, _ = self.time_bin(absolute_minutes)
        base = max(1.0, float(distance_miles) / max(MIN_SPEED_MPH, self._speed(absolute_minutes)) * 60.0)
        return base * self._duration_calibration_multiplier(origin, destination, low, distance_miles)

    def node_metric(self, zone_id: int, absolute_minutes: float) -> Dict[str, float]:
        zone_id = int(zone_id)
        low, high, frac = self.time_bin(absolute_minutes)
        a, b = self._node_map.get((low, zone_id)), self._node_map.get((high, zone_id))
        if a is None and b is None:
            return {
                "pickup_probability_15m": 0.0, "expected_wait_min": 75.0,
                "pickup_rate_index": 0.0, "expected_trip_revenue": 0.0,
                "expected_trip_duration_min": 15.0, "expected_trip_distance_miles": 2.0,
                "confidence": MIN_CONFIDENCE, "latent_vacant_supply_index": 0.0,
                "competition_ratio": 10.0, "raw_pickups_per_bin_day": 0.0,
                "raw_dropoffs_per_bin_day": 0.0, "estimated_competitors_equivalent": 0.0,
            }
        if a is None: a = b
        if b is None: b = a
        out = {k: (1-frac)*float(a[k]) + frac*float(b[k]) for k in a}
        wait = out[f"expected_wait_min_{self.competition_scenario}"]
        out["expected_wait_min"] = wait
        out["estimated_competitors_equivalent"] = out.get(
            f"estimated_competitors_equivalent_{self.competition_scenario}",
            out.get("estimated_competitors_equivalent", 0.0),
        )
        out["pickup_probability_15m"] = 1.0 - math.exp(-TIME_BIN_MINUTES / max(1e-6, wait))
        return out

    def estimate_od(self, origin: int, destination: int, start_time: str | float) -> Dict[str, float | str]:
        """Return a finite estimate for every valid origin-destination-time query.

        Sources, in order: time-specific EB row, full-history OD adjusted by time-of-day,
        and spatial/gravity fallback. Results are cached by 15-minute cell.
        """
        origin, destination = int(origin), int(destination)
        if origin not in self.zone_to_index or destination not in self.zone_to_index:
            raise ValueError(f"Unknown taxi zone pair: {origin}->{destination}")
        absolute = parse_clock_time(start_time)
        low, high, frac = self.time_bin(absolute)
        a = self._estimate_od_bin(origin, destination, low)
        b = self._estimate_od_bin(origin, destination, high)
        numeric = ["expected_driver_revenue", "expected_duration_minutes", "expected_distance_miles",
                   "revenue_std", "duration_std", "distance_std", "confidence", "observed_trip_count"]
        result: Dict[str, float | str] = {
            "origin": origin, "destination": destination,
            "start_time": format_clock_time(absolute),
            "estimate_source": a["estimate_source"] if frac < 0.5 else b["estimate_source"],
        }
        for k in numeric:
            result[k] = (1-frac)*float(a[k]) + frac*float(b[k])
        result["is_directly_observed"] = bool(float(result["observed_trip_count"]) > 0)
        return result

    @staticmethod
    def _circular_bin_distance(a: int, b: int) -> int:
        d = abs(int(a) - int(b))
        return min(d, 96 - d)

    @staticmethod
    def _weighted(values: Sequence[float], weights: Sequence[float], default: float) -> float:
        pairs = [(float(v), float(w)) for v, w in zip(values, weights) if np.isfinite(v) and float(w) > 0]
        if not pairs:
            return float(default)
        return float(np.average([v for v, _ in pairs], weights=[w for _, w in pairs]))

    def _spatial_global_prior(self, origin: int, destination: int) -> Optional[Dict[str, float]]:
        values: List[Dict[str, float]] = []
        weights: List[float] = []
        for neigh, miles in self._nearest_zones.get(origin, []):
            row = self._global_od_map.get((neigh, destination))
            if row is not None:
                values.append(row); weights.append(float(row["trip_count"]) * math.exp(-miles / SPATIAL_DECAY_MILES))
        for neigh, miles in self._nearest_zones.get(destination, []):
            row = self._global_od_map.get((origin, neigh))
            if row is not None:
                values.append(row); weights.append(float(row["trip_count"]) * math.exp(-miles / SPATIAL_DECAY_MILES))
        if not values or sum(weights) <= 0:
            return None
        return {
            "revenue": self._weighted([x["revenue"] for x in values], weights, self._global_rpm),
            "duration_min": self._weighted([x["duration_min"] for x in values], weights, 15.0),
            "distance_miles": self._weighted([x["distance_miles"] for x in values], weights, 2.0),
            "revenue_std": self._weighted([x["revenue_std"] for x in values], weights, 2.0),
            "duration_std": self._weighted([x["duration_std"] for x in values], weights, 3.0),
            "distance_std": self._weighted([x["distance_std"] for x in values], weights, 0.5),
            "effective_weight": float(min(sum(weights), 100.0)),
        }

    def _estimate_od_bin(self, origin: int, destination: int, tb: int) -> Dict[str, float | str]:
        key = (int(tb), int(origin), int(destination))
        if key in self._full_od_cache:
            return dict(self._full_od_cache[key])

        direct = self._od_pair_map.get(key)
        if direct is not None:
            result: Dict[str, float | str] = {
                "expected_driver_revenue": direct["revenue"],
                "expected_duration_minutes": direct["duration_min"],
                "expected_distance_miles": direct["distance_miles"],
                "revenue_std": direct["revenue_std"],
                "duration_std": direct["duration_std"],
                "distance_std": direct["distance_std"],
                "confidence": direct["confidence"],
                "observed_trip_count": direct["observed_trip_count"],
                "estimate_source": direct["estimate_source"],
            }
        else:
            absolute = tb * TIME_BIN_MINUTES
            observed = self._observed_od_map.get(key)
            global_row = self._global_od_map.get((origin, destination))
            speed_ratio = self._global_speed / self._speed(absolute)
            rpm_ratio = self._rpm(absolute) / max(0.1, self._global_rpm)

            if observed is not None:
                n = float(observed["trip_count"])
                if global_row is None:
                    global_rev = observed["revenue"]
                    global_dur = observed["duration_min"]
                    global_dist = observed["distance_miles"]
                    global_rev_std = observed["revenue_std"]
                    global_dur_std = observed["duration_std"]
                    global_dist_std = observed["distance_std"]
                else:
                    global_rev = global_row["revenue"] * rpm_ratio
                    global_dur = global_row["duration_min"] * speed_ratio
                    global_dist = global_row["distance_miles"]
                    global_rev_std = global_row["revenue_std"]
                    global_dur_std = global_row["duration_std"] * speed_ratio
                    global_dist_std = global_row["distance_std"]
                reliability = n / (n + OD_OBS_PRIOR_STRENGTH)
                result = {
                    "expected_driver_revenue": reliability * observed["revenue"] + (1-reliability) * global_rev,
                    "expected_duration_minutes": reliability * observed["duration_min"] + (1-reliability) * global_dur,
                    "expected_distance_miles": reliability * observed["distance_miles"] + (1-reliability) * global_dist,
                    "revenue_std": max(0.5, reliability*observed["revenue_std"] + (1-reliability)*global_rev_std),
                    "duration_std": max(1.0, reliability*observed["duration_std"] + (1-reliability)*global_dur_std),
                    "distance_std": max(0.1, reliability*observed["distance_std"] + (1-reliability)*global_dist_std),
                    "confidence": float(np.clip(n/(n+OD_OBS_PRIOR_STRENGTH), MIN_CONFIDENCE, 0.995)),
                    "observed_trip_count": n,
                    "estimate_source": "full_observed_dense_eb" if n >= 20 else "full_observed_sparse_eb",
                }
            else:
                temporal_rows = []
                temporal_weights = []
                for row in self._observed_pair_rows.get((origin, destination), []):
                    dist = self._circular_bin_distance(tb, int(row["time_bin"]))
                    if 0 < dist <= TEMPORAL_WINDOW_BINS:
                        w = float(row["trip_count"]) * math.exp(-dist / TEMPORAL_DECAY_BINS)
                        temporal_rows.append(row); temporal_weights.append(w)
                if temporal_rows and sum(temporal_weights) > 0:
                    temporal_count = float(sum(temporal_weights))
                    default_rev = global_row["revenue"]*rpm_ratio if global_row is not None else self._global_rpm*2.0
                    default_dur = global_row["duration_min"]*speed_ratio if global_row is not None else 15.0
                    default_dist = global_row["distance_miles"] if global_row is not None else 2.0
                    result = {
                        "expected_driver_revenue": self._weighted([r["revenue"] for r in temporal_rows], temporal_weights, default_rev) * rpm_ratio,
                        "expected_duration_minutes": self._weighted([r["duration_min"] for r in temporal_rows], temporal_weights, default_dur) * speed_ratio,
                        "expected_distance_miles": self._weighted([r["distance_miles"] for r in temporal_rows], temporal_weights, default_dist),
                        "revenue_std": max(0.5, self._weighted([r["revenue_std"] for r in temporal_rows], temporal_weights, 2.0)),
                        "duration_std": max(1.0, self._weighted([r["duration_std"] for r in temporal_rows], temporal_weights, 3.0) * speed_ratio),
                        "distance_std": max(0.1, self._weighted([r["distance_std"] for r in temporal_rows], temporal_weights, 0.5)),
                        "confidence": float(np.clip(0.15 + 0.70*temporal_count/(temporal_count+30.0), MIN_CONFIDENCE, 0.88)),
                        "observed_trip_count": 0.0,
                        "estimate_source": "temporal_same_od_eb",
                    }
                elif global_row is not None:
                    n = float(global_row["trip_count"])
                    result = {
                        "expected_driver_revenue": max(0.0, global_row["revenue"] * rpm_ratio),
                        "expected_duration_minutes": max(1.0, global_row["duration_min"] * speed_ratio),
                        "expected_distance_miles": max(0.01, global_row["distance_miles"]),
                        "revenue_std": max(0.5, global_row["revenue_std"]),
                        "duration_std": max(1.0, global_row["duration_std"] * speed_ratio),
                        "distance_std": max(0.1, global_row["distance_std"]),
                        "confidence": float(np.clip(0.20 + 0.70*n/(n+50.0), MIN_CONFIDENCE, 0.90)),
                        "observed_trip_count": 0.0,
                        "estimate_source": "global_od_time_adjusted",
                    }
                else:
                    route = self.route(origin, destination, absolute_minutes=absolute)
                    distance = max(0.01, float(route["distance_miles"]))
                    duration = max(1.0, float(route["duration_min"]))
                    intercept = float(self._gravity.get("intercept", 2.5))
                    beta_d = float(self._gravity.get("distance", 2.5))
                    beta_t = float(self._gravity.get("duration", 0.2))
                    gravity_revenue = max(2.5, intercept + beta_d*distance + beta_t*duration)
                    spatial = self._spatial_global_prior(origin, destination)
                    if spatial is None:
                        revenue = 0.55*gravity_revenue + 0.45*self._rpm(absolute)*distance
                        result = {
                            "expected_driver_revenue": revenue,
                            "expected_duration_minutes": duration,
                            "expected_distance_miles": distance,
                            "revenue_std": max(1.0, 0.35*revenue),
                            "duration_std": max(2.0, 0.30*duration),
                            "distance_std": max(0.25, 0.25*distance),
                            "confidence": float(np.clip(0.08 + 0.17*float(route["confidence"]), MIN_CONFIDENCE, 0.28)),
                            "observed_trip_count": 0.0,
                            "estimate_source": "spatial_gravity_time_fallback",
                        }
                    else:
                        spatial_duration = spatial["duration_min"] * speed_ratio
                        revenue = 0.55*spatial["revenue"]*rpm_ratio + 0.45*gravity_revenue
                        duration_mix = 0.55*spatial_duration + 0.45*duration
                        distance_mix = 0.55*spatial["distance_miles"] + 0.45*distance
                        result = {
                            "expected_driver_revenue": max(2.5, revenue),
                            "expected_duration_minutes": max(1.0, duration_mix),
                            "expected_distance_miles": max(0.01, distance_mix),
                            "revenue_std": max(1.0, spatial["revenue_std"]),
                            "duration_std": max(2.0, spatial["duration_std"]*speed_ratio),
                            "distance_std": max(0.25, spatial["distance_std"]),
                            "confidence": float(np.clip(0.12 + 0.35*spatial["effective_weight"]/(spatial["effective_weight"]+30.0), MIN_CONFIDENCE, 0.48)),
                            "observed_trip_count": 0.0,
                            "estimate_source": "spatial_neighbor_gravity_eb",
                        }

        # Physical consistency guards. They prevent very long trips from being assigned
        # unrealistically short durations after shrinkage.
        distance = max(0.01, float(result["expected_distance_miles"]))
        minimum_duration = distance / MAX_SPEED_MPH * 60.0
        calibrated_floor = 0.85 * self.calibrated_structural_duration(origin, destination, tb * TIME_BIN_MINUTES, distance)
        result["expected_duration_minutes"] = max(
            1.0, minimum_duration, calibrated_floor, float(result["expected_duration_minutes"])
        )
        result["duration_calibrated_floor_minutes"] = float(calibrated_floor)
        result["expected_driver_revenue"] = max(0.0, float(result["expected_driver_revenue"]))
        result["confidence"] = float(np.clip(float(result["confidence"]), MIN_CONFIDENCE, 0.995))

        if len(self._full_od_cache) >= FULL_OD_CACHE_MAX_SIZE:
            self._full_od_cache.clear()
        self._full_od_cache[key] = dict(result)
        return result

    def od_options(self, origin: int, absolute_minutes: float) -> List[Dict[str, float]]:
        origin = int(origin)
        low, high, frac = self.time_bin(absolute_minutes)
        low_rows, high_rows = self._od_map.get((low, origin), []), self._od_map.get((high, origin), [])
        by_dest: Dict[int, Dict[str, float]] = {}
        for weight, rows in ((1-frac, low_rows), (frac, high_rows)):
            for row in rows:
                d = int(row["destination"])
                rec = by_dest.setdefault(d, {"destination": d, "probability": 0.0, "metric_weight": 0.0,
                    "revenue": 0.0, "duration_min": 0.0, "distance_miles": 0.0,
                    "revenue_std": 0.0, "duration_std": 0.0, "distance_std": 0.0, "confidence": 0.0})
                pw = weight * float(row["probability"])
                rec["probability"] += pw; rec["metric_weight"] += pw
                for k in ("revenue", "duration_min", "distance_miles", "revenue_std", "duration_std", "distance_std", "confidence"):
                    rec[k] += pw*float(row[k])
        if not by_dest:
            # Rare defensive fallback: construct options from global origin flows.
            choices = self._global_origin_lists.get(origin, [])[:12]
            if not choices:
                choices = [(z, 1.0) for z in self.zone_ids[:12]]
            total = sum(c for _, c in choices) or 1.0
            rows = []
            for d, c in choices:
                est = self.estimate_od(origin, d, absolute_minutes)
                rows.append({"destination": int(d), "probability": float(c/total),
                    "revenue": float(est["expected_driver_revenue"]),
                    "duration_min": float(est["expected_duration_minutes"]),
                    "distance_miles": float(est["expected_distance_miles"]),
                    "revenue_std": float(est["revenue_std"]), "duration_std": float(est["duration_std"]),
                    "distance_std": float(est["distance_std"]), "confidence": float(est["confidence"])})
            return rows
        rows = list(by_dest.values())
        total_p = sum(max(0.0, r["probability"]) for r in rows)
        for rec in rows:
            mw = max(rec.pop("metric_weight"), 1e-12)
            for k in ("revenue", "duration_min", "distance_miles", "revenue_std", "duration_std", "distance_std", "confidence"):
                rec[k] /= mw
            rec["probability"] = max(0.0, rec["probability"]) / max(1e-12, total_p)
        rows.sort(key=lambda x: x["probability"], reverse=True)
        return rows

    def route(self, origin: int, destination: int, method: str = "dijkstra", absolute_minutes: float | None = None) -> Dict[str, object]:
        origin, destination = int(origin), int(destination)
        absolute = 12*60.0 if absolute_minutes is None else float(absolute_minutes)
        tb = self.time_bin(absolute)[0]
        cache_key = (origin, destination, tb, method)
        if cache_key in self._route_cache:
            return dict(self._route_cache[cache_key])
        if origin == destination:
            return {"origin": origin, "destination": destination, "path": [origin], "duration_min": 0.0,
                    "distance_miles": 0.0, "confidence": 1.0, "time_bin": tb}
        try:
            if method == "astar":
                x2, y2 = self.centroids_xy[destination]
                def heuristic(node: int, _target: int) -> float:
                    x1, y1 = self.centroids_xy[int(node)]
                    return math.hypot(x1-x2, y1-y2)/5280.0 / MAX_SPEED_MPH * 60.0
                path = nx.astar_path(self._reposition_graph, origin, destination, heuristic=heuristic, weight="duration_min")
            else:
                path = nx.shortest_path(self._reposition_graph, origin, destination, weight="duration_min")
            base_duration, distance, confs = 0.0, 0.0, []
            for a, b in zip(path[:-1], path[1:]):
                edge = self._reposition_graph[a][b]
                base_duration += float(edge["duration_min"]); distance += float(edge["distance_miles"])
                confs.append(float(edge.get("confidence", 1.0)))
            duration = base_duration * self._global_speed / self._speed(absolute)
            result = {"origin": origin, "destination": destination, "path": [int(x) for x in path],
                      "duration_min": max(0.1, duration), "distance_miles": distance,
                      "confidence": float(np.mean(confs) if confs else 1.0), "time_bin": tb}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            x1, y1 = self.centroids_xy[origin]; x2, y2 = self.centroids_xy[destination]
            miles = math.hypot(x1-x2, y1-y2)/5280.0
            result = {"origin": origin, "destination": destination, "path": [origin, destination],
                      "duration_min": max(0.1, miles/self._speed(absolute)*60.0),
                      "distance_miles": miles, "confidence": 0.15, "time_bin": tb}
        self._route_cache[cache_key] = dict(result)
        return result

    def k_shortest_routes(self, origin: int, destination: int, k: int = 3, absolute_minutes: float | None = None) -> List[Dict[str, object]]:
        if origin == destination:
            return [self.route(origin, destination, absolute_minutes=absolute_minutes)]
        try:
            generator = nx.shortest_simple_paths(self._reposition_graph, int(origin), int(destination), weight="duration_min")
            out = []
            for path in generator:
                distance = sum(float(self._reposition_graph[a][b]["distance_miles"]) for a,b in zip(path[:-1], path[1:]))
                base = sum(float(self._reposition_graph[a][b]["duration_min"]) for a,b in zip(path[:-1], path[1:]))
                confs = [float(self._reposition_graph[a][b].get("confidence",1.0)) for a,b in zip(path[:-1], path[1:])]
                absolute = 12*60.0 if absolute_minutes is None else float(absolute_minutes)
                out.append({"origin": int(origin), "destination": int(destination), "path": [int(x) for x in path],
                            "duration_min": base*self._global_speed/self._speed(absolute), "distance_miles": distance,
                            "confidence": float(np.mean(confs) if confs else 1.0)})
                if len(out) >= int(k): break
            return out
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [self.route(origin, destination, absolute_minutes=absolute_minutes)]

    def candidate_zone_details(self, start_zone: int, start_time: str | float, n: int = POLICY_CANDIDATE_ZONES) -> pd.DataFrame:
        """Diverse candidate generator independent of any single policy formula."""
        absolute = parse_clock_time(start_time)
        cache_key = (int(start_zone), self.time_bin(absolute)[0], int(n))
        records = []
        dest_prob = {int(o["destination"]): float(o["probability"]) for o in self.od_options(start_zone, absolute)}
        tb = self.time_bin(absolute)[0]
        pool: List[int] = []
        for zone, _distance in self._candidate_nearest_zones.get(int(start_zone), []):
            if zone not in pool:
                pool.append(int(zone))
        for zone in self._candidate_leaders_by_bin.get(int(tb), []):
            if zone not in pool:
                pool.append(int(zone))
        for zone, _probability in sorted(dest_prob.items(), key=lambda item: item[1], reverse=True)[:max(8, int(n))]:
            if zone not in pool:
                pool.append(int(zone))
        for zone, _count in self._global_origin_lists.get(int(start_zone), [])[:max(8, int(n))]:
            if zone not in pool:
                pool.append(int(zone))
        for z in pool:
            if z == int(start_zone):
                continue
            route = self.route(start_zone, z, absolute_minutes=absolute)
            if route["duration_min"] > MAX_REPOSITION_MINUTES or route["distance_miles"] > MAX_REPOSITION_MILES:
                continue
            arrival = absolute + float(route["duration_min"])
            m = self.node_metric(z, arrival)
            net = m["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE*m["expected_trip_distance_miles"]
            cycle = max(1.0, m["expected_wait_min"] + m["expected_trip_duration_min"])
            records.append({
                "zone_id": z, "route_minutes": float(route["duration_min"]), "route_miles": float(route["distance_miles"]),
                "demand": m["pickup_rate_index"], "net_trip": net, "net_rate": net/cycle,
                "competition_advantage": 1.0/max(1e-6,m["competition_ratio"]),
                "proximity": 1.0/max(1.0,float(route["duration_min"])),
                "passenger_destination_probability": dest_prob.get(z,0.0), "confidence": m["confidence"],
            })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        metrics = ["demand","net_trip","net_rate","competition_advantage","proximity","passenger_destination_probability"]
        for metric in metrics:
            df[f"rank_{metric}"] = df[metric].rank(method="min", ascending=False)
        # Union of category leaders prevents the candidate set from being pre-filtered by
        # the same objective later used by GreedyNetEarningsPolicy.
        selected: List[int] = []
        quota = max(1, min(DYNAMIC_CANDIDATE_CATEGORY_QUOTA, int(n)))
        for metric in metrics:
            for z in df.nlargest(quota, metric)["zone_id"].astype(int):
                if z not in selected: selected.append(z)
        df["rank_sum"] = df[[f"rank_{m}" for m in metrics]].sum(axis=1)
        for z in df.sort_values(["rank_sum","route_minutes"])["zone_id"].astype(int):
            if z not in selected: selected.append(z)
            if len(selected) >= int(n): break
        order = {z:i for i,z in enumerate(selected[:int(n)])}
        df["selected"] = df["zone_id"].isin(order)
        df["selected_rank"] = df["zone_id"].map(order).fillna(10**6).astype(int)+1
        return df.sort_values(["selected_rank","rank_sum"]).reset_index(drop=True)

    def dynamic_candidate_zones(self, start_zone: int, start_time: str | float, n: int = POLICY_CANDIDATE_ZONES, exclude_start: bool = True) -> List[int]:
        absolute = parse_clock_time(start_time)
        key = (int(start_zone), self.time_bin(absolute)[0], int(n))
        if key not in self._candidate_cache:
            df = self.candidate_zone_details(start_zone, absolute, n)
            self._candidate_cache[key] = df[df["selected"]].sort_values("selected_rank")["zone_id"].astype(int).head(int(n)).tolist() if not df.empty else []
        out = list(self._candidate_cache[key])
        return [z for z in out if not exclude_start or z != int(start_zone)]

    # Backward-compatible name used by validation scripts.
    def candidate_zones(self, start_zone: int, start_time: str | float, n: int = POLICY_CANDIDATE_ZONES, exclude_start: bool = True) -> List[int]:
        return self.dynamic_candidate_zones(start_zone, start_time, n, exclude_start)

    def with_min_od_confidence(self, threshold: float) -> "DynamicTaxiEnvironment":
        clone = self.clone()
        filtered = clone.od_metrics[clone.od_metrics["confidence"] >= float(threshold)].copy()
        retained = []
        for (tb, origin), group in clone.od_metrics.groupby(["time_bin", "origin"]):
            g = filtered[(filtered["time_bin"]==tb)&(filtered["origin"]==origin)].copy()
            if g.empty: g = group.nlargest(1,"confidence").copy()
            total = float(g["destination_probability"].sum())
            if total > 0: g["destination_probability"] /= total
            retained.append(g)
        clone.od_metrics = pd.concat(retained, ignore_index=True) if retained else filtered
        clone._build_indexes(); return clone

    def with_origin_revenue_multiplier(self, zone_id: int, multiplier: float) -> "DynamicTaxiEnvironment":
        clone = self.clone()
        zone_id, multiplier = int(zone_id), float(multiplier)
        mask = clone.od_metrics["origin"].astype(int) == zone_id
        clone.od_metrics.loc[mask, "avg_driver_revenue"] *= multiplier
        node_mask = clone.node_metrics["zone_id"].astype(int) == zone_id
        clone.node_metrics.loc[node_mask, "expected_trip_revenue"] *= multiplier
        if not clone.global_od_metrics.empty:
            gm = clone.global_od_metrics["origin"].astype(int) == zone_id
            clone.global_od_metrics.loc[gm, "avg_driver_revenue"] *= multiplier
        if not clone.observed_od_metrics.empty:
            om = clone.observed_od_metrics["origin"].astype(int) == zone_id
            clone.observed_od_metrics.loc[om, "observed_avg_driver_revenue"] *= multiplier
        clone._build_indexes()
        return clone

    def sample_trip_metrics(self, option: Mapping[str,float], randomness: ScenarioRandomness, event_index: int) -> Tuple[float,float,float]:
        def draw(mean: float, std: float, name: str, floor: float) -> float:
            mean, std = max(float(mean),floor), max(float(std),0.0)
            return mean if std <= 1e-9 else max(floor, mean + randomness.normal(event_index,name)*std)
        return (draw(option["revenue"],option.get("revenue_std",0.0),"trip_revenue",0.0),
                draw(option["duration_min"],option.get("duration_std",0.0),"trip_duration",1.0),
                draw(option["distance_miles"],option.get("distance_std",0.0),"trip_distance",0.01))

    def simulate_episode(
        self, policy: object, start_zone: int, start_time: str, scenario_id: int, base_seed: int,
        allowed_reposition_zones: Optional[Sequence[int]] = None,
        horizon_minutes: int = HORIZON_MINUTES, record_events: bool = True,
        dynamic_candidate_count: int = POLICY_CANDIDATE_ZONES,
    ) -> EpisodeResult:
        """Simulate one complete operating window with auditable minute accounting.

        Completed-trip revenue is counted only when the passenger trip finishes within the
        horizon. If a sampled trip crosses the boundary, the in-window occupied portion is
        tracked separately as unfinished_occupied_minutes. Every episode satisfies
        waiting + empty + occupied + unfinished occupied + terminal unused = horizon.
        """
        start_abs = parse_clock_time(start_time)
        zone, elapsed = int(start_zone), 0.0
        gross = cost = waiting = empty_minutes = occupied_minutes = 0.0
        empty_miles = occupied_miles = unfinished_occupied = terminal_unused = 0.0
        completed = unfinished = wait_decisions = reposition_decisions = 0
        first_action = ""
        events: List[TrajectoryEvent] = []
        randomness = ScenarioRandomness(scenario_id, base_seed)
        decision_index = 0
        trajectory_index = 0
        fixed_allowed = None if allowed_reposition_zones is None else tuple(
            dict.fromkeys(int(z) for z in allowed_reposition_zones)
        )

        def add_event(event_type: str, from_zone: int, to_zone: int, start: float, end: float, **kwargs: object) -> None:
            nonlocal trajectory_index
            if record_events:
                events.append(TrajectoryEvent(
                    trajectory_index, event_type, int(from_zone), int(to_zone),
                    float(start), float(end),
                    format_clock_time(start_abs + start), format_clock_time(start_abs + end),
                    **kwargs,
                ))
            trajectory_index += 1

        while elapsed < horizon_minutes - 1e-9:
            remaining = horizon_minutes - elapsed
            absolute = start_abs + elapsed
            if fixed_allowed is None:
                allowed = self.dynamic_candidate_zones(zone, absolute, n=dynamic_candidate_count)
            else:
                allowed = [
                    z for z in fixed_allowed
                    if z != zone and self.route(zone, z, absolute_minutes=absolute)["duration_min"] < remaining
                ]
            state = TaxiState(zone, elapsed, remaining)
            action = policy.select_action(self, state, start_abs, allowed)
            if action != WAIT_ACTION:
                try:
                    target = int(action)
                except (TypeError, ValueError):
                    target = zone
                if target == zone or target not in set(allowed):
                    action = WAIT_ACTION
            if not first_action:
                first_action = str(action)

            if action == WAIT_ACTION:
                wait_decisions += 1
                metric = self.node_metric(zone, absolute)
                mean_wait = max(0.1, float(metric["expected_wait_min"]))
                sampled = -mean_wait * math.log(max(1e-12, randomness.uniform(decision_index, "wait_time")))

                if sampled > WAIT_REEVALUATION_MINUTES:
                    dt = min(float(WAIT_REEVALUATION_MINUTES), remaining)
                    add_event(
                        "WAIT_NO_PICKUP", zone, zone, elapsed, elapsed + dt,
                        wait_minutes=dt, confidence=metric["confidence"], selected_action=WAIT_ACTION,
                    )
                    waiting += dt
                    elapsed += dt
                    decision_index += 1
                    continue

                wait_dt = min(sampled, remaining)
                if wait_dt > 1e-9:
                    add_event(
                        "WAIT_FOR_PICKUP", zone, zone, elapsed, elapsed + wait_dt,
                        wait_minutes=wait_dt, confidence=metric["confidence"], selected_action=WAIT_ACTION,
                    )
                waiting += wait_dt
                pickup_elapsed = elapsed + wait_dt
                if pickup_elapsed >= horizon_minutes - 1e-9:
                    elapsed = float(horizon_minutes)
                    decision_index += 1
                    break

                options = self.od_options(zone, start_abs + pickup_elapsed)
                if not options:
                    elapsed = pickup_elapsed
                    decision_index += 1
                    continue
                u = randomness.uniform(decision_index, "destination")
                cumulative = 0.0
                selected = options[-1]
                for option in options:
                    cumulative += float(option["probability"])
                    if u <= cumulative:
                        selected = option
                        break
                revenue, duration, distance = self.sample_trip_metrics(selected, randomness, decision_index)
                destination = int(selected["destination"])
                trip_end = pickup_elapsed + duration
                if trip_end > horizon_minutes + 1e-9:
                    unfinished = 1
                    in_window = max(0.0, horizon_minutes - pickup_elapsed)
                    unfinished_occupied += in_window
                    add_event(
                        "UNFINISHED_TRIP", zone, destination, pickup_elapsed, horizon_minutes,
                        distance_miles=distance * (in_window / max(duration, 1e-9)),
                        confidence=float(selected["confidence"]), completed_within_horizon=False,
                        selected_action=WAIT_ACTION,
                    )
                    elapsed = float(horizon_minutes)
                    decision_index += 1
                    break

                trip_cost = OCCUPIED_COST_PER_MILE * distance
                add_event(
                    "OCCUPIED_TRIP", zone, destination, pickup_elapsed, trip_end,
                    revenue=revenue, operating_cost=trip_cost, distance_miles=distance,
                    confidence=float(selected["confidence"]), selected_action=WAIT_ACTION,
                )
                gross += revenue
                cost += trip_cost
                completed += 1
                occupied_minutes += duration
                occupied_miles += distance
                elapsed = trip_end
                zone = destination
                decision_index += 1
                continue

            reposition_decisions += 1
            target = int(action)
            route = self.route(zone, target, absolute_minutes=absolute)
            duration = float(route["duration_min"])
            distance = float(route["distance_miles"])
            if duration <= 1e-9:
                # Defensive progress guard for malformed routes.
                dt = min(float(WAIT_REEVALUATION_MINUTES), remaining)
                add_event("TERMINAL_OR_INVALID_ROUTE_WAIT", zone, zone, elapsed, elapsed + dt,
                          wait_minutes=dt, selected_action=str(target))
                waiting += dt
                elapsed += dt
                decision_index += 1
                continue
            if duration > remaining + 1e-9:
                terminal_unused += remaining
                add_event("TERMINAL_UNUSED", zone, zone, elapsed, horizon_minutes, selected_action=str(target))
                elapsed = float(horizon_minutes)
                decision_index += 1
                break
            reposition_cost = EMPTY_COST_PER_MILE * distance
            add_event(
                "EMPTY_REPOSITION", zone, target, elapsed, elapsed + duration,
                operating_cost=reposition_cost, distance_miles=distance,
                confidence=float(route["confidence"]), route_path="|".join(map(str, route["path"])),
                selected_action=str(target),
            )
            cost += reposition_cost
            empty_minutes += duration
            empty_miles += distance
            elapsed += duration
            zone = target
            decision_index += 1

        accounted_without_terminal = waiting + empty_minutes + occupied_minutes + unfinished_occupied
        residual = max(0.0, float(horizon_minutes) - accounted_without_terminal - terminal_unused)
        if residual > 1e-7:
            start_unused = max(0.0, float(horizon_minutes) - residual)
            add_event("TERMINAL_UNUSED", zone, zone, start_unused, float(horizon_minutes), selected_action="END")
            terminal_unused += residual
        accounted = waiting + empty_minutes + occupied_minutes + unfinished_occupied + terminal_unused
        accounting_error = float(horizon_minutes) - accounted

        return EpisodeResult(
            algorithm=str(getattr(policy, "name", policy.__class__.__name__)),
            scenario_id=int(scenario_id), start_zone=int(start_zone), start_time=str(start_time),
            horizon_minutes=int(horizon_minutes), gross_revenue=float(gross), operating_cost=float(cost),
            net_earnings=float(gross - cost), completed_trips=int(completed),
            waiting_minutes=float(waiting), empty_minutes=float(empty_minutes),
            occupied_minutes=float(occupied_minutes), empty_miles=float(empty_miles),
            occupied_miles=float(occupied_miles), unfinished_trip_at_horizon=int(unfinished),
            end_zone=int(zone), unfinished_occupied_minutes=float(unfinished_occupied),
            terminal_unused_minutes=float(terminal_unused), accounted_minutes=float(accounted),
            time_accounting_error=float(accounting_error), wait_decisions=int(wait_decisions),
            reposition_decisions=int(reposition_decisions), first_action=first_action or WAIT_ACTION,
            events=events,
        )


class FiniteHorizonModel:
    """Expected finite-horizon semi-Markov model used by DP and Dynamic Zone Shapley."""

    def __init__(self, env: DynamicTaxiEnvironment, start_time: str | float, horizon_minutes: int,
                 candidate_zones: Optional[Sequence[int]] = None,
                 dynamic_candidate_count: int = POLICY_CANDIDATE_ZONES) -> None:
        self.env=env; self.start_abs=parse_clock_time(start_time); self.horizon_minutes=int(horizon_minutes)
        self.dt=float(DECISION_TIME_STEP_MINUTES); self.steps=int(math.ceil(horizon_minutes/self.dt))
        self.candidate_zones = None if candidate_zones is None else tuple(dict.fromkeys(int(z) for z in candidate_zones))
        self.dynamic_candidate_count=int(dynamic_candidate_count)
        self.zones=env.zone_ids; self.zone_to_idx=env.zone_to_index; self.n_zones=len(self.zones)
        self._prepare_wait_transitions(); self._prepare_reposition_transitions()

    @staticmethod
    def _conditional_exponential_wait(mean_wait: float, interval: float) -> float:
        mean_wait=max(1e-6,float(mean_wait)); e=math.exp(-interval/mean_wait)
        return float(np.clip(mean_wait-interval*e/max(1e-12,1-e),0.0,interval))

    def _prepare_wait_transitions(self) -> None:
        self.wait_data=[]
        for step in range(self.steps):
            absolute=self.start_abs+step*self.dt; per=[]
            for zone in self.zones:
                node=self.env.node_metric(zone,absolute); mean=node["expected_wait_min"]
                p=1.0-math.exp(-self.dt/max(1e-6,mean)); cw=self._conditional_exponential_wait(mean,self.dt)
                trans=[]
                for option in self.env.od_options(zone,absolute+cw):
                    total=cw+float(option["duration_min"]); end_pos=step+total/self.dt
                    reward=float(option["revenue"])-OCCUPIED_COST_PER_MILE*float(option["distance_miles"])
                    trans.append((self.zone_to_idx[int(option["destination"])],float(option["probability"]),float(end_pos),reward))
                per.append({"pickup_prob":p,"transitions":trans})
            self.wait_data.append(per)

    def _prepare_reposition_transitions(self) -> None:
        self.reposition_data: List[List[List[Tuple[int,float,float,float]]]]=[]
        for step in range(self.steps):
            absolute=self.start_abs+step*self.dt; per_zone=[]
            for origin in self.zones:
                candidates = list(self.candidate_zones) if self.candidate_zones is not None else self.env.dynamic_candidate_zones(origin,absolute,n=self.dynamic_candidate_count)
                rows=[]
                for target in candidates:
                    if target==origin: continue
                    route=self.env.route(origin,target,absolute_minutes=absolute); duration=float(route["duration_min"]); distance=float(route["distance_miles"])
                    if duration>MAX_REPOSITION_MINUTES or distance>MAX_REPOSITION_MILES: continue
                    rows.append((int(target),float(step+duration/self.dt),float(EMPTY_COST_PER_MILE*distance),float(route["confidence"])))
                per_zone.append(rows)
            self.reposition_data.append(per_zone)

    def _future(self,value: np.ndarray,end_pos: float,zi: int) -> float:
        if end_pos > self.steps+1e-12: return 0.0
        lo=int(math.floor(end_pos)); hi=int(math.ceil(end_pos))
        lo=min(max(lo,0),self.steps); hi=min(max(hi,0),self.steps)
        if lo==hi: return float(value[lo,zi])
        frac=end_pos-lo
        return float((1-frac)*value[lo,zi]+frac*value[hi,zi])

    def solve(self, allowed_zone_ids: Optional[Sequence[int]], base_allowed_zone_ids: Sequence[int]=(), return_policy: bool=False) -> Tuple[np.ndarray,Optional[np.ndarray]]:
        allowed_set=None if allowed_zone_ids is None else (set(map(int,allowed_zone_ids))|set(map(int,base_allowed_zone_ids)))
        value=np.zeros((self.steps+1,self.n_zones),float)
        policy=np.full((self.steps,self.n_zones),-1,dtype=np.int32) if return_policy else None
        for step in range(self.steps-1,-1,-1):
            for zi in range(self.n_zones):
                data=self.wait_data[step][zi]; p=float(data["pickup_prob"])
                wait_value=(1-p)*value[step+1,zi]
                pickup=0.0
                for dest_idx,prob,end_pos,reward in data["transitions"]:
                    if end_pos<=self.steps+1e-12: pickup += prob*(reward+self._future(value,end_pos,dest_idx))
                wait_value += p*pickup
                best=wait_value; best_target=-1
                for target,end_pos,reposition_cost,_conf in self.reposition_data[step][zi]:
                    if allowed_set is not None and target not in allowed_set: continue
                    if end_pos>self.steps+1e-12: continue
                    candidate=-reposition_cost+self._future(value,end_pos,self.zone_to_idx[target])
                    if candidate>best+1e-12: best=candidate; best_target=target
                value[step,zi]=best
                if policy is not None: policy[step,zi]=best_target
        return value,policy

    def start_value(self,start_zone: int,allowed_zone_ids: Optional[Sequence[int]],base_allowed_zone_ids: Sequence[int]=()) -> float:
        v,_=self.solve(allowed_zone_ids,base_allowed_zone_ids,False)
        return float(v[0,self.zone_to_idx[int(start_zone)]])


class DynamicProgramPolicy:
    name="finite_horizon_value_iteration"
    def __init__(self, model: FiniteHorizonModel, allowed_zone_ids: Optional[Sequence[int]]=None, base_allowed_zone_ids: Sequence[int]=()) -> None:
        self.model=model; self.allowed_zone_ids=None if allowed_zone_ids is None else tuple(map(int,allowed_zone_ids))
        _,p=model.solve(self.allowed_zone_ids,base_allowed_zone_ids,True)
        if p is None: raise RuntimeError("Policy table was not created")
        self.policy_table=p
    def select_action(self,env: DynamicTaxiEnvironment,state: TaxiState,start_absolute_minutes: float,allowed_zones: Sequence[int]) -> str|int:
        step=min(self.model.steps-1,max(0,int(state.elapsed_minutes//self.model.dt)))
        target=int(self.policy_table[step,self.model.zone_to_idx[int(state.zone_id)]])
        return WAIT_ACTION if target<0 or target not in set(map(int,allowed_zones)) else target
