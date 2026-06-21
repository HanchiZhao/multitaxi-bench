"""
Route / opportunity environment for the revised single-driver taxi-zone project.

This version stabilizes the environment around three layers from the uploaded notes:

Stage 1. Static graph algorithms
    - Dijkstra shortest-time / shortest-distance search
    - A* shortest-time / shortest-distance search with geographic heuristic
    - Yen-style k-shortest simple paths through NetworkX shortest_simple_paths
    - Utility-based candidate path ranking without using negative utility as graph cost

Stage 2. Offline dynamic policy baselines
    - Finite-horizon value iteration on discrete states (zone, time_bin)
    - Tabular Q-learning with epsilon-greedy exploration

Stage 3. DQN preparation
    - A stable exporter for state/action/reward transition records.
      Full neural DQN training is intentionally kept outside this file because it
      requires PyTorch and careful tuning.

Stage 4. Shapley preparation
    - Candidate opportunity nodes
    - Candidate path pool with path masks and route utility
    - These are the inputs needed by a later Shapley module.

Data required in data/:
- taxi_zones.shp/.shx/.dbf/.prj/.cpg
- taxi_zone_lookup.csv
- yellow_tripdata_2025-01.parquet ... yellow_tripdata_2025-06.parquet

Run from project root:
    python scripts\route_environment.py
"""

from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
MONTH_TAG = f"{MONTHS[0]}_{MONTHS[-1]}"

ZONE_SHP = os.path.join(DATA_DIR, "taxi_zones.shp")
ZONE_LOOKUP = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")

TRIP_USE_COLS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
]

# Cleaning thresholds
MIN_FARE = 2.50
MAX_FARE = 500.0
MIN_DISTANCE_MILES = 0.01
MAX_DISTANCE_MILES = 100.0
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 240.0

# Model assumptions
DRIVER_SHARE = 0.70
FALLBACK_SPEED_MPH = 12.0
ASTAR_HEURISTIC_SPEED_MPH = 60.0  # deliberately high so time heuristic is conservative
SPATIAL_ADJACENCY_DISTANCE_FEET = 200.0
MIN_OD_TRIP_COUNT = 5

# Recommendation / route utility weights.
DEFAULT_INCOME_WEIGHT = 1.0
DEFAULT_DEMAND_WEIGHT = 0.30
DEFAULT_TRAVEL_TIME_WEIGHT = 0.25
DEFAULT_DISTANCE_WEIGHT = 0.05

# Revised single-driver problem: one origin and a finite planning horizon.
DEFAULT_TIME_BUDGET_MIN = 120.0
DEFAULT_TIME_BIN_MIN = 15.0
DEFAULT_GAMMA = 0.97

# RL defaults
DEFAULT_Q_EPISODES = 1500
DEFAULT_Q_ALPHA = 0.20
DEFAULT_Q_EPSILON_START = 1.00
DEFAULT_Q_EPSILON_END = 0.05


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_read_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path, columns=columns)


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or math.isclose(float(hi), float(lo)):
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def feet_to_miles(feet: float) -> float:
    return float(feet) / 5280.0


def miles_to_minutes(miles: float, speed_mph: float = FALLBACK_SPEED_MPH) -> float:
    if speed_mph <= 0:
        raise ValueError("speed_mph must be positive")
    return float(miles) / speed_mph * 60.0


def path_signature(path: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in path)


def deduplicate_path_results(paths: Iterable["PathResult"]) -> List["PathResult"]:
    seen: Dict[Tuple[int, ...], PathResult] = {}
    for p in paths:
        seen[path_signature(p.path)] = p
    return list(seen.values())


# =============================================================================
# Data construction
# =============================================================================

def load_zone_geometries(shapefile_path: str = ZONE_SHP, lookup_path: str = ZONE_LOOKUP) -> gpd.GeoDataFrame:
    """Load TLC zone polygons and join human-readable names."""
    zones = gpd.read_file(shapefile_path)
    lookup = pd.read_csv(lookup_path)

    if "LocationID" not in zones.columns:
        possible = [c for c in zones.columns if c.lower() == "locationid"]
        if not possible:
            raise ValueError("Cannot find LocationID in taxi_zones shapefile.")
        zones = zones.rename(columns={possible[0]: "LocationID"})

    zones["LocationID"] = zones["LocationID"].astype(int)
    lookup["LocationID"] = lookup["LocationID"].astype(int)

    zones = zones[zones.geometry.notna()].copy()
    zones = zones[zones.geometry.is_valid].copy()

    keep_lookup_cols = [c for c in ["LocationID", "Borough", "Zone", "service_zone"] if c in lookup.columns]
    zones = zones.merge(lookup[keep_lookup_cols], on="LocationID", how="left", suffixes=("", "_lookup"))

    if "Zone" not in zones.columns and "zone" in zones.columns:
        zones = zones.rename(columns={"zone": "Zone"})
    if "Borough" not in zones.columns and "borough" in zones.columns:
        zones = zones.rename(columns={"borough": "Borough"})

    zones = zones.sort_values("LocationID").reset_index(drop=True)
    return zones


def build_nodes_table(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    centroids = zones.geometry.centroid
    nodes = pd.DataFrame(
        {
            "zone_id": zones["LocationID"].astype(int),
            "borough": zones.get("Borough", pd.Series([None] * len(zones))).astype(str),
            "zone_name": zones.get("Zone", pd.Series([None] * len(zones))).astype(str),
            "service_zone": zones.get("service_zone", pd.Series([None] * len(zones))).astype(str),
            "centroid_x": centroids.x,
            "centroid_y": centroids.y,
            "area_sqft": zones.geometry.area,
        }
    )
    return nodes


def build_spatial_adjacency_edges(
    zones: gpd.GeoDataFrame,
    distance_threshold_feet: float = SPATIAL_ADJACENCY_DISTANCE_FEET,
) -> pd.DataFrame:
    """Build directed adjacency edges from polygon touches / near touches."""
    zone_ids = zones["LocationID"].astype(int).tolist()
    geom_by_id = dict(zip(zone_ids, zones.geometry))
    centroid_by_id = dict(zip(zone_ids, zones.geometry.centroid))

    sindex = zones.sindex
    records: List[Dict[str, float]] = []
    seen_pairs = set()

    for idx, row in zones.iterrows():
        u = int(row["LocationID"])
        geom = row.geometry
        candidate_idx = list(sindex.query(geom.buffer(distance_threshold_feet), predicate="intersects"))
        for j in candidate_idx:
            if j == idx:
                continue
            v = int(zones.iloc[j]["LocationID"])
            pair = tuple(sorted((u, v)))
            if pair in seen_pairs:
                continue
            g2 = geom_by_id[v]
            dist_feet = float(geom.distance(g2))
            if geom.touches(g2) or geom.intersects(g2) or dist_feet <= distance_threshold_feet:
                seen_pairs.add(pair)
                c1 = centroid_by_id[u]
                c2 = centroid_by_id[v]
                centroid_miles = feet_to_miles(c1.distance(c2))
                fallback_time = miles_to_minutes(centroid_miles)
                for a, b in [(u, v), (v, u)]:
                    records.append(
                        {
                            "origin": int(a),
                            "destination": int(b),
                            "edge_source": "spatial_adjacency",
                            "trip_count": 0,
                            "avg_distance_miles": centroid_miles,
                            "avg_duration_min": fallback_time,
                            "avg_driver_income": 0.0,
                        }
                    )

    return pd.DataFrame(records)


def clean_trip_data(df: pd.DataFrame, valid_zone_ids: set[int], month: str) -> pd.DataFrame:
    """Clean raw yellow taxi trip records for node-income and OD estimates."""
    missing = [c for c in TRIP_USE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {month}: {missing}")

    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    for c in ["PULocationID", "DOLocationID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
        ]
    )

    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)
    df = df[df["PULocationID"].isin(valid_zone_ids) & df["DOLocationID"].isin(valid_zone_ids)]

    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[
        (df["fare_amount"].between(MIN_FARE, MAX_FARE))
        & (df["trip_distance"].between(MIN_DISTANCE_MILES, MAX_DISTANCE_MILES))
        & (df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN))
    ].copy()

    df["tip_amount"] = df["tip_amount"].fillna(0.0)
    df["total_amount"] = df["total_amount"].fillna(df["fare_amount"] + df["tip_amount"])
    df["driver_income"] = DRIVER_SHARE * (df["fare_amount"] + df["tip_amount"])

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["time_bin"] = df["pickup_hour"] * 4 + (df["tpep_pickup_datetime"].dt.minute // 15)
    df["month"] = month
    return df


def load_and_clean_all_trips(months: Sequence[str], valid_zone_ids: set[int]) -> pd.DataFrame:
    frames = []
    for month in months:
        path = os.path.join(DATA_DIR, f"yellow_tripdata_{month}.parquet")
        raw = safe_read_parquet(path, columns=TRIP_USE_COLS)
        clean = clean_trip_data(raw, valid_zone_ids, month)
        print(f"{month}: raw {len(raw):,} -> clean {len(clean):,}")
        frames.append(clean)
    return pd.concat(frames, ignore_index=True)


def build_node_income_table(trips: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Estimate pickup-zone opportunity statistics from cleaned trips."""
    if trips.empty:
        raise ValueError("No trips available to build node income table.")

    total_hours = max(
        1.0,
        (trips["tpep_pickup_datetime"].max() - trips["tpep_pickup_datetime"].min()).total_seconds() / 3600.0,
    )

    pickup = (
        trips.groupby("PULocationID")
        .agg(
            pickup_count=("PULocationID", "size"),
            avg_driver_income_per_trip=("driver_income", "mean"),
            median_driver_income_per_trip=("driver_income", "median"),
            avg_trip_distance_miles=("trip_distance", "mean"),
            avg_trip_duration_min=("duration_min", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_tip=("tip_amount", "mean"),
            avg_total_amount=("total_amount", "mean"),
        )
        .reset_index()
        .rename(columns={"PULocationID": "zone_id"})
    )

    dropoff = (
        trips.groupby("DOLocationID")
        .agg(dropoff_count=("DOLocationID", "size"))
        .reset_index()
        .rename(columns={"DOLocationID": "zone_id"})
    )

    out = nodes.merge(pickup, on="zone_id", how="left").merge(dropoff, on="zone_id", how="left")
    numeric_fill_zero = [
        "pickup_count",
        "dropoff_count",
        "avg_driver_income_per_trip",
        "median_driver_income_per_trip",
        "avg_trip_distance_miles",
        "avg_trip_duration_min",
        "avg_fare",
        "avg_tip",
        "avg_total_amount",
    ]
    for c in numeric_fill_zero:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    out["pickup_rate_per_hour"] = out["pickup_count"] / total_hours
    out["dropoff_rate_per_hour"] = out["dropoff_count"] / total_hours
    out["expected_pickups_2h"] = out["pickup_rate_per_hour"] * 2.0
    out["expected_income_2h_proxy"] = out["avg_driver_income_per_trip"] * out["expected_pickups_2h"]

    out["income_score"] = normalize_series(out["avg_driver_income_per_trip"])
    out["demand_score"] = normalize_series(out["pickup_rate_per_hour"])
    out["opportunity_score"] = normalize_series(out["expected_income_2h_proxy"])

    return out.sort_values("zone_id").reset_index(drop=True)


def build_od_weights(trips: pd.DataFrame) -> pd.DataFrame:
    od = (
        trips.groupby(["PULocationID", "DOLocationID"])
        .agg(
            trip_count=("PULocationID", "size"),
            avg_distance_miles=("trip_distance", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_driver_income=("driver_income", "mean"),
            avg_total_amount=("total_amount", "mean"),
        )
        .reset_index()
        .rename(columns={"PULocationID": "origin", "DOLocationID": "destination"})
    )
    od = od[od["trip_count"] >= MIN_OD_TRIP_COUNT].copy()
    od["edge_source"] = "historical_od"
    return od


def build_edges(spatial_edges: pd.DataFrame, od_weights: pd.DataFrame) -> pd.DataFrame:
    """Merge spatial fallback edges and historical OD edges.

    Historical OD edges take priority when duplicate origin-destination pairs exist.
    Spatial edges preserve local graph connectivity when no historical OD edge exists.
    """
    edge_cols = [
        "origin",
        "destination",
        "edge_source",
        "trip_count",
        "avg_distance_miles",
        "avg_duration_min",
        "avg_driver_income",
    ]
    s = spatial_edges[edge_cols].copy()
    o = od_weights[edge_cols].copy()
    s["priority"] = 0
    o["priority"] = 1
    edges = pd.concat([s, o], ignore_index=True)
    edges = edges.sort_values(["origin", "destination", "priority"]).drop_duplicates(
        ["origin", "destination"], keep="last"
    )
    edges = edges.drop(columns="priority")

    edges["travel_time_cost"] = edges["avg_duration_min"].clip(lower=0.1)
    edges["distance_cost"] = edges["avg_distance_miles"].clip(lower=0.001)

    # This is only a diagnostic score, not a graph search cost.
    edges["edge_income_score"] = normalize_series(edges["avg_driver_income"])
    edges["edge_time_score"] = normalize_series(edges["avg_duration_min"])
    edges["edge_distance_score"] = normalize_series(edges["avg_distance_miles"])
    edges["edge_utility_proxy"] = (
        edges["edge_income_score"]
        - DEFAULT_TRAVEL_TIME_WEIGHT * edges["edge_time_score"]
        - DEFAULT_DISTANCE_WEIGHT * edges["edge_distance_score"]
    )
    return edges.sort_values(["origin", "destination"]).reset_index(drop=True)


def build_networkx_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        attrs = row.to_dict()
        node_id = int(attrs.pop("zone_id"))
        G.add_node(node_id, **attrs)
    for _, row in edges.iterrows():
        attrs = row.to_dict()
        u = int(attrs.pop("origin"))
        v = int(attrs.pop("destination"))
        G.add_edge(u, v, **attrs)
    return G


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class PathResult:
    path: List[int]
    total_travel_time: float
    total_distance: float
    total_expected_income: float
    total_demand_score: float
    route_utility: float
    num_edges: int
    algorithm: str = "unknown"
    destination: Optional[int] = None


@dataclass
class NodeRecommendation:
    origin: int
    destination: int
    travel_time_min: float
    distance_miles: float
    avg_driver_income_per_trip: float
    pickup_rate_per_hour: float
    expected_income_2h_proxy: float
    recommendation_utility: float
    zone_name: str
    borough: str
    algorithm: str = "utility"


@dataclass
class TransitionRecord:
    state_zone: int
    state_time_bin: int
    action_zone: int
    next_zone: int
    next_time_bin: int
    reward: float
    travel_time_min: float
    distance_miles: float
    done: bool


# =============================================================================
# Environment class
# =============================================================================

class TaxiOpportunityEnvironment:
    """Single-driver opportunity environment.

    Main task: given one start zone and a time budget, rank where a driver should
    move to seek future pickups. It also provides static graph algorithms,
    tabular RL baselines, and Shapley-ready candidate paths.
    """

    def __init__(self, graph: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame):
        self.G = graph
        self.nodes = nodes.copy()
        self.edges = edges.copy()
        self.node_info = self.nodes.set_index("zone_id").to_dict(orient="index")
        self.edge_lookup = self.edges.set_index(["origin", "destination"]).to_dict(orient="index")
        self._spatial_graph: Optional[nx.Graph] = None

    # -------------------------------------------------------------------------
    # Basic accessors
    # -------------------------------------------------------------------------

    def validate_zone(self, zone_id: int) -> None:
        if int(zone_id) not in self.G.nodes:
            raise ValueError(f"Invalid zone_id {zone_id}. Not found in graph.")

    def get_node_expected_income(self, zone_id: int) -> Dict[str, float]:
        self.validate_zone(zone_id)
        return self.node_info[int(zone_id)]

    def edge_data(self, u: int, v: int) -> Dict:
        if self.G.has_edge(u, v):
            return self.G[int(u)][int(v)]
        raise ValueError(f"No edge from {u} to {v}")

    def _spatial_only_graph(self) -> nx.Graph:
        if self._spatial_graph is not None:
            return self._spatial_graph
        SG = nx.Graph()
        SG.add_nodes_from(self.G.nodes)
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_source") == "spatial_adjacency":
                SG.add_edge(int(u), int(v))
        self._spatial_graph = SG
        return SG

    def centroid_distance_miles(self, u: int, v: int) -> float:
        ui = self.node_info[int(u)]
        vi = self.node_info[int(v)]
        dx = float(ui["centroid_x"]) - float(vi["centroid_x"])
        dy = float(ui["centroid_y"]) - float(vi["centroid_y"])
        return feet_to_miles(math.sqrt(dx * dx + dy * dy))

    # -------------------------------------------------------------------------
    # Unified utility
    # -------------------------------------------------------------------------

    def recommendation_utility(
        self,
        origin: int,
        destination: int,
        travel_time_min: float,
        distance_miles: float,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        uniform_income: bool = False,
        income_weight: float = DEFAULT_INCOME_WEIGHT,
        demand_weight: float = DEFAULT_DEMAND_WEIGHT,
        travel_time_weight: float = DEFAULT_TRAVEL_TIME_WEIGHT,
        distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
    ) -> float:
        """Unified utility for moving from origin to a candidate opportunity zone."""
        del origin  # kept for future origin-specific extensions
        info = self.node_info[int(destination)]
        income_score = 1.0 if uniform_income else float(info.get("income_score", 0.0))
        demand_score = float(info.get("demand_score", 0.0))
        t_norm = float(travel_time_min) / max(1.0, float(time_budget_min))
        d_norm = float(distance_miles) / 30.0 if np.isfinite(distance_miles) else 0.0
        return (
            income_weight * income_score
            + demand_weight * demand_score
            - travel_time_weight * t_norm
            - distance_weight * d_norm
        )

    def compute_path_result(
        self,
        path: Sequence[int],
        algorithm: str = "unknown",
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        uniform_income: bool = False,
        income_weight: float = DEFAULT_INCOME_WEIGHT,
        demand_weight: float = DEFAULT_DEMAND_WEIGHT,
        travel_time_weight: float = DEFAULT_TRAVEL_TIME_WEIGHT,
        distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
    ) -> PathResult:
        if len(path) < 2:
            raise ValueError("Path must contain at least origin and destination.")
        total_time = 0.0
        total_distance = 0.0
        total_demand_score = 0.0
        for u, v in zip(path[:-1], path[1:]):
            d = self.edge_data(int(u), int(v))
            total_time += float(d.get("avg_duration_min", 0.0))
            total_distance += float(d.get("avg_distance_miles", 0.0))
            total_demand_score += float(self.node_info[int(v)].get("demand_score", 0.0))

        origin = int(path[0])
        dest = int(path[-1])
        dest_info = self.node_info[dest]
        expected_income = float(dest_info.get("avg_driver_income_per_trip", 0.0))
        route_utility = self.recommendation_utility(
            origin=origin,
            destination=dest,
            travel_time_min=total_time,
            distance_miles=total_distance,
            time_budget_min=time_budget_min,
            uniform_income=uniform_income,
            income_weight=income_weight,
            demand_weight=demand_weight,
            travel_time_weight=travel_time_weight,
            distance_weight=distance_weight,
        )
        return PathResult(
            path=[int(x) for x in path],
            total_travel_time=total_time,
            total_distance=total_distance,
            total_expected_income=expected_income,
            total_demand_score=total_demand_score,
            route_utility=route_utility,
            num_edges=len(path) - 1,
            algorithm=algorithm,
            destination=dest,
        )

    # -------------------------------------------------------------------------
    # Static graph algorithms: Dijkstra / A* / Yen KSP
    # -------------------------------------------------------------------------

    def _astar_heuristic(self, target: int, weight: str):
        target = int(target)

        def h(n1: int, n2: int = target) -> float:
            miles = self.centroid_distance_miles(int(n1), int(n2))
            if weight == "distance_cost":
                return miles
            # Conservative time heuristic. High speed avoids overestimation.
            return miles_to_minutes(miles, ASTAR_HEURISTIC_SPEED_MPH)

        return h

    def shortest_path(
        self,
        origin: int,
        destination: int,
        weight: str = "travel_time_cost",
        algorithm: str = "dijkstra",
        **kwargs,
    ) -> PathResult:
        self.validate_zone(origin)
        self.validate_zone(destination)
        origin = int(origin)
        destination = int(destination)
        algorithm = algorithm.lower()
        if algorithm == "astar":
            path = nx.astar_path(
                self.G,
                origin,
                destination,
                heuristic=self._astar_heuristic(destination, weight),
                weight=weight,
            )
            alg_name = f"astar_{'distance' if weight == 'distance_cost' else 'time'}"
        elif algorithm == "dijkstra":
            path = nx.shortest_path(self.G, origin, destination, weight=weight)
            alg_name = f"dijkstra_{'distance' if weight == 'distance_cost' else 'time'}"
        else:
            raise ValueError("algorithm must be 'dijkstra' or 'astar'")
        return self.compute_path_result(path, algorithm=alg_name, **kwargs)

    def shortest_time_path(self, origin: int, destination: int, algorithm: str = "dijkstra", **kwargs) -> PathResult:
        return self.shortest_path(origin, destination, weight="travel_time_cost", algorithm=algorithm, **kwargs)

    def shortest_distance_path(self, origin: int, destination: int, algorithm: str = "dijkstra", **kwargs) -> PathResult:
        return self.shortest_path(origin, destination, weight="distance_cost", algorithm=algorithm, **kwargs)

    def yen_k_shortest_paths(
        self,
        origin: int,
        destination: int,
        k: int = 5,
        weight: str = "travel_time_cost",
        max_path_len: Optional[int] = None,
        algorithm_label: Optional[str] = None,
        **kwargs,
    ) -> List[PathResult]:
        """Yen-style K shortest simple paths via NetworkX shortest_simple_paths.

        NetworkX implements the standard deviation/spur-path idea for simple paths.
        The edge weight is non-negative travel time or distance, not negative utility.
        """
        self.validate_zone(origin)
        self.validate_zone(destination)
        results: List[PathResult] = []
        label = algorithm_label or f"yen_ksp_{'distance' if weight == 'distance_cost' else 'time'}"
        try:
            gen = nx.shortest_simple_paths(self.G, int(origin), int(destination), weight=weight)
            for path in gen:
                if max_path_len is not None and len(path) > max_path_len:
                    continue
                results.append(self.compute_path_result(path, algorithm=label, **kwargs))
                if len(results) >= k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return results

    # Backward-compatible alias.
    def k_shortest_paths(self, *args, **kwargs) -> List[PathResult]:
        return self.yen_k_shortest_paths(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Static recommendation algorithms
    # -------------------------------------------------------------------------

    def _reachable_lengths(self, origin: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        self.validate_zone(origin)
        lengths_time = nx.single_source_dijkstra_path_length(self.G, int(origin), weight="travel_time_cost")
        lengths_dist = nx.single_source_dijkstra_path_length(self.G, int(origin), weight="distance_cost")
        return lengths_time, lengths_dist

    def rank_opportunity_nodes(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        top_k: int = 20,
        candidate_nodes: Optional[Iterable[int]] = None,
        uniform_income: bool = False,
        algorithm: str = "utility",
        income_weight: float = DEFAULT_INCOME_WEIGHT,
        demand_weight: float = DEFAULT_DEMAND_WEIGHT,
        travel_time_weight: float = DEFAULT_TRAVEL_TIME_WEIGHT,
        distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
    ) -> pd.DataFrame:
        """Rank destination zones by several stable algorithms.

        algorithm options:
        - 'utility': unified opportunity utility
        - 'dijkstra_time': shortest travel time to zone
        - 'dijkstra_distance': shortest distance to zone
        - 'highest_income': destination avg income score / value
        - 'highest_demand': destination demand score
        """
        self.validate_zone(origin)
        origin = int(origin)
        if candidate_nodes is None:
            candidate_nodes = list(self.G.nodes)
        candidate_nodes = [int(z) for z in candidate_nodes if int(z) != origin and int(z) in self.G.nodes]

        lengths_time, lengths_dist = self._reachable_lengths(origin)
        rows = []
        for z in candidate_nodes:
            t = float(lengths_time.get(z, np.inf))
            if not np.isfinite(t) or t > time_budget_min:
                continue
            d = float(lengths_dist.get(z, np.nan))
            info = self.node_info[z]
            income_score = 1.0 if uniform_income else float(info.get("income_score", 0.0))
            demand_score = float(info.get("demand_score", 0.0))
            utility = self.recommendation_utility(
                origin=origin,
                destination=z,
                travel_time_min=t,
                distance_miles=d,
                time_budget_min=time_budget_min,
                uniform_income=uniform_income,
                income_weight=income_weight,
                demand_weight=demand_weight,
                travel_time_weight=travel_time_weight,
                distance_weight=distance_weight,
            )
            rows.append(
                {
                    "origin": origin,
                    "destination": z,
                    "zone_name": info.get("zone_name", ""),
                    "borough": info.get("borough", ""),
                    "travel_time_min": t,
                    "distance_miles": d,
                    "avg_driver_income_per_trip": float(info.get("avg_driver_income_per_trip", 0.0)),
                    "pickup_rate_per_hour": float(info.get("pickup_rate_per_hour", 0.0)),
                    "expected_income_2h_proxy": float(info.get("expected_income_2h_proxy", 0.0)),
                    "income_score": income_score,
                    "demand_score": demand_score,
                    "opportunity_score": float(info.get("opportunity_score", 0.0)),
                    "recommendation_utility": utility,
                    "algorithm": algorithm,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out

        algorithm = algorithm.lower()
        if algorithm in {"utility", "utility_rank"}:
            out = out.sort_values("recommendation_utility", ascending=False)
        elif algorithm in {"dijkstra_time", "shortest_time", "time"}:
            out = out.sort_values(["travel_time_min", "recommendation_utility"], ascending=[True, False])
        elif algorithm in {"dijkstra_distance", "shortest_distance", "distance"}:
            out = out.sort_values(["distance_miles", "recommendation_utility"], ascending=[True, False])
        elif algorithm in {"highest_income", "income"}:
            out = out.sort_values(["avg_driver_income_per_trip", "recommendation_utility"], ascending=[False, False])
        elif algorithm in {"highest_demand", "demand"}:
            out = out.sort_values(["pickup_rate_per_hour", "recommendation_utility"], ascending=[False, False])
        else:
            raise ValueError(f"Unknown ranking algorithm: {algorithm}")
        out["algorithm"] = algorithm
        return out.head(top_k).reset_index(drop=True)

    def candidate_opportunity_nodes(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        max_candidates: int = 80,
        m_per_algorithm: int = 30,
        uniform_income: bool = False,
        algorithms: Sequence[str] = ("utility", "dijkstra_time", "dijkstra_distance", "highest_income", "highest_demand"),
    ) -> List[int]:
        """Candidate destination nodes for later Shapley / algorithm comparison.

        This is now multi-seed rather than one single route corridor. It collects
        high-ranking zones from several stable algorithms, then uses utility and
        O-D accessibility to cap the player set.
        """
        selected: Dict[int, float] = {}
        for alg in algorithms:
            ranked = self.rank_opportunity_nodes(
                origin=origin,
                time_budget_min=time_budget_min,
                top_k=m_per_algorithm,
                uniform_income=uniform_income,
                algorithm=alg,
            )
            if ranked.empty:
                continue
            for _, row in ranked.iterrows():
                z = int(row["destination"])
                # Keep best utility score if a node appears in multiple algorithms.
                selected[z] = max(float(row["recommendation_utility"]), selected.get(z, -np.inf))
        ordered = sorted(selected.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in ordered[:max_candidates]]

    def build_candidate_path_pool(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        candidate_nodes: Optional[Sequence[int]] = None,
        m_per_family: int = 5,
        final_k: int = 30,
        uniform_income: bool = False,
        diversity_threshold: float = 0.85,
    ) -> List[PathResult]:
        """Build a static candidate path pool using the uploaded-note logic.

        Families included:
        1. shortest-time destinations and paths
        2. shortest-distance destinations and paths
        3. highest-income destinations with shortest-time paths
        4. highest-demand destinations with shortest-time paths
        5. utility-ranked destinations with both time and distance paths

        Then all paths are scored by the same unified utility and the top diverse
        paths are retained. We never search directly with negative utility cost,
        avoiding negative-cycle instability.
        """
        if candidate_nodes is None:
            candidate_nodes = self.candidate_opportunity_nodes(
                origin=origin,
                time_budget_min=time_budget_min,
                max_candidates=80,
                uniform_income=uniform_income,
            )
        candidate_set = set(int(z) for z in candidate_nodes)

        families = ["dijkstra_time", "dijkstra_distance", "highest_income", "highest_demand", "utility"]
        destination_pool: List[int] = []
        for alg in families:
            top = self.rank_opportunity_nodes(
                origin=origin,
                time_budget_min=time_budget_min,
                top_k=m_per_family,
                candidate_nodes=candidate_set,
                uniform_income=uniform_income,
                algorithm=alg,
            )
            if not top.empty:
                destination_pool.extend(top["destination"].astype(int).tolist())

        destination_pool = list(dict.fromkeys(destination_pool))  # preserve order, remove duplicates
        path_candidates: List[PathResult] = []
        for dest in destination_pool:
            try:
                path_candidates.append(
                    self.shortest_time_path(
                        origin,
                        dest,
                        algorithm="dijkstra",
                        time_budget_min=time_budget_min,
                        uniform_income=uniform_income,
                    )
                )
            except Exception:
                pass
            try:
                path_candidates.append(
                    self.shortest_distance_path(
                        origin,
                        dest,
                        algorithm="dijkstra",
                        time_budget_min=time_budget_min,
                        uniform_income=uniform_income,
                    )
                )
            except Exception:
                pass
            # Yen KSP gives extra local alternatives for a few important destinations.
            path_candidates.extend(
                self.yen_k_shortest_paths(
                    origin,
                    dest,
                    k=2,
                    weight="travel_time_cost",
                    algorithm_label="yen_ksp_time",
                    time_budget_min=time_budget_min,
                    uniform_income=uniform_income,
                )
            )

        # Filter by time budget and sort by unified utility.
        unique = deduplicate_path_results(path_candidates)
        feasible = [p for p in unique if p.total_travel_time <= time_budget_min]
        feasible.sort(key=lambda p: p.route_utility, reverse=True)

        # Diversity filter: avoid keeping many nearly identical node sequences.
        selected_paths: List[PathResult] = []
        selected_node_sets: List[set[int]] = []
        for pr in feasible:
            node_set = set(pr.path)
            too_similar = False
            for existing in selected_node_sets:
                union_size = len(node_set | existing)
                overlap = len(node_set & existing) / union_size if union_size else 0.0
                if overlap >= diversity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected_paths.append(pr)
                selected_node_sets.append(node_set)
            if len(selected_paths) >= final_k:
                break

        # If diversity filter was too strict, fill remaining slots by utility.
        if len(selected_paths) < final_k:
            seen = {path_signature(p.path) for p in selected_paths}
            for pr in feasible:
                sig = path_signature(pr.path)
                if sig in seen:
                    continue
                selected_paths.append(pr)
                seen.add(sig)
                if len(selected_paths) >= final_k:
                    break

        return selected_paths[:final_k]

    def shapley_ready_path_table(
        self,
        origin: int,
        paths: Sequence[PathResult],
        baseline_path: Optional[Sequence[int]] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """Convert candidate paths into a table useful for later Shapley.

        Players are optional nodes that appear in candidate paths but not in the
        baseline path. Each path receives a bitmask of required optional nodes.
        A later Shapley module can evaluate v(S) by checking which path masks are
        subsets of a coalition mask.
        """
        if baseline_path is None:
            if paths:
                baseline_path = paths[0].path
            else:
                baseline_path = [int(origin)]
        baseline_nodes = set(int(z) for z in baseline_path)
        optional_nodes = sorted({int(z) for p in paths for z in p.path if int(z) not in baseline_nodes})
        node_to_bit = {z: i for i, z in enumerate(optional_nodes)}

        records = []
        for path_id, pr in enumerate(paths):
            mask = 0
            required = []
            for z in pr.path:
                z = int(z)
                if z in node_to_bit:
                    mask |= 1 << node_to_bit[z]
                    required.append(z)
            records.append(
                {
                    "path_id": path_id,
                    "algorithm": pr.algorithm,
                    "destination": int(pr.path[-1]),
                    "path": "-".join(str(x) for x in pr.path),
                    "required_optional_nodes": ",".join(str(x) for x in required),
                    "required_mask": mask,
                    "travel_time_min": pr.total_travel_time,
                    "distance_miles": pr.total_distance,
                    "dest_expected_income": pr.total_expected_income,
                    "route_utility": pr.route_utility,
                }
            )
        return pd.DataFrame(records), node_to_bit

    # -------------------------------------------------------------------------
    # MDP / RL helpers
    # -------------------------------------------------------------------------

    def available_actions(
        self,
        zone: int,
        spatial_only: bool = True,
        include_stay: bool = True,
        candidate_nodes: Optional[set[int]] = None,
        max_actions: Optional[int] = None,
    ) -> List[int]:
        """Actions are next zones. Default uses one-hop spatial neighbors + stay.

        This keeps the dynamic problem small and stable. Historical OD edges are
        dense, so they are not used by default as actions.
        """
        zone = int(zone)
        if spatial_only:
            SG = self._spatial_only_graph()
            actions = [int(v) for v in SG.neighbors(zone)] if zone in SG else []
        else:
            actions = [int(v) for v in self.G.successors(zone)]
        if candidate_nodes is not None:
            actions = [a for a in actions if a in candidate_nodes]
        if include_stay and zone not in actions:
            actions.append(zone)
        if max_actions is not None and len(actions) > max_actions:
            # Keep most promising actions by immediate reward proxy.
            actions = sorted(actions, key=lambda a: self._action_reward(zone, a, DEFAULT_TIME_BIN_MIN)[0], reverse=True)[
                :max_actions
            ]
        return actions

    def _move_stats(self, zone: int, action_zone: int, default_step_min: float = DEFAULT_TIME_BIN_MIN) -> Tuple[float, float]:
        zone = int(zone)
        action_zone = int(action_zone)
        if zone == action_zone:
            return float(default_step_min), 0.0
        if self.G.has_edge(zone, action_zone):
            d = self.G[zone][action_zone]
            return float(d.get("avg_duration_min", default_step_min)), float(d.get("avg_distance_miles", 0.0))
        # Fallback if a spatial neighbor somehow lacks directed edge.
        miles = self.centroid_distance_miles(zone, action_zone)
        return miles_to_minutes(miles), miles

    def _action_reward(
        self,
        zone: int,
        action_zone: int,
        default_step_min: float = DEFAULT_TIME_BIN_MIN,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        uniform_income: bool = False,
    ) -> Tuple[float, float, float]:
        travel_time, distance = self._move_stats(zone, action_zone, default_step_min)
        reward = self.recommendation_utility(
            origin=zone,
            destination=action_zone,
            travel_time_min=travel_time,
            distance_miles=distance,
            time_budget_min=time_budget_min,
            uniform_income=uniform_income,
        )
        return reward, travel_time, distance

    def finite_horizon_value_iteration(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        bin_minutes: float = DEFAULT_TIME_BIN_MIN,
        gamma: float = DEFAULT_GAMMA,
        candidate_nodes: Optional[Sequence[int]] = None,
        spatial_only_actions: bool = True,
        uniform_income: bool = False,
        max_actions_per_state: Optional[int] = 12,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
        """Finite-horizon value iteration / backward dynamic programming.

        State: (zone, time_bin). Action: move to a neighboring zone or stay.
        Transition: deterministic to action_zone after one or more time bins.
        Reward: immediate opportunity utility at the action zone minus travel cost.
        """
        self.validate_zone(origin)
        horizon_bins = int(math.ceil(time_budget_min / bin_minutes))
        if candidate_nodes is None:
            candidate_nodes = self.candidate_opportunity_nodes(origin, time_budget_min, max_candidates=80)
            candidate_nodes = list(set(candidate_nodes) | {int(origin)})
        state_nodes = set(int(z) for z in candidate_nodes if int(z) in self.G.nodes)
        state_nodes.add(int(origin))

        V: Dict[Tuple[int, int], float] = {(z, horizon_bins): 0.0 for z in state_nodes}
        policy: Dict[Tuple[int, int], int] = {}

        for tbin in range(horizon_bins - 1, -1, -1):
            for z in state_nodes:
                actions = self.available_actions(
                    z,
                    spatial_only=spatial_only_actions,
                    include_stay=True,
                    candidate_nodes=state_nodes,
                    max_actions=max_actions_per_state,
                )
                best_value = -np.inf
                best_action = z
                for a in actions:
                    reward, travel_time, _distance = self._action_reward(
                        z,
                        a,
                        default_step_min=bin_minutes,
                        time_budget_min=time_budget_min,
                        uniform_income=uniform_income,
                    )
                    step_bins = max(1, int(math.ceil(travel_time / bin_minutes)))
                    next_t = min(horizon_bins, tbin + step_bins)
                    value = reward + gamma * V.get((int(a), next_t), 0.0)
                    if value > best_value:
                        best_value = value
                        best_action = int(a)
                V[(int(z), tbin)] = float(best_value)
                policy[(int(z), tbin)] = int(best_action)

        records = []
        for (z, tbin), a in sorted(policy.items()):
            reward, travel_time, distance = self._action_reward(
                z, a, default_step_min=bin_minutes, time_budget_min=time_budget_min, uniform_income=uniform_income
            )
            records.append(
                {
                    "zone": z,
                    "time_bin": tbin,
                    "minutes_elapsed": tbin * bin_minutes,
                    "best_action_zone": a,
                    "best_action_name": self.node_info.get(a, {}).get("zone_name", ""),
                    "value": V[(z, tbin)],
                    "immediate_reward": reward,
                    "action_travel_time_min": travel_time,
                    "action_distance_miles": distance,
                }
            )
        return pd.DataFrame(records), V, policy

    def tabular_q_learning(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        bin_minutes: float = DEFAULT_TIME_BIN_MIN,
        gamma: float = DEFAULT_GAMMA,
        episodes: int = DEFAULT_Q_EPISODES,
        alpha: float = DEFAULT_Q_ALPHA,
        epsilon_start: float = DEFAULT_Q_EPSILON_START,
        epsilon_end: float = DEFAULT_Q_EPSILON_END,
        candidate_nodes: Optional[Sequence[int]] = None,
        spatial_only_actions: bool = True,
        uniform_income: bool = False,
        seed: int = 42,
        max_actions_per_state: Optional[int] = 12,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[int, int, int], float]]:
        """Stable tabular Q-learning baseline with epsilon-greedy exploration."""
        rng = random.Random(seed)
        self.validate_zone(origin)
        horizon_bins = int(math.ceil(time_budget_min / bin_minutes))
        if candidate_nodes is None:
            candidate_nodes = self.candidate_opportunity_nodes(origin, time_budget_min, max_candidates=80)
            candidate_nodes = list(set(candidate_nodes) | {int(origin)})
        state_nodes = set(int(z) for z in candidate_nodes if int(z) in self.G.nodes)
        state_nodes.add(int(origin))

        Q: Dict[Tuple[int, int, int], float] = {}

        def q_value(z: int, tbin: int, a: int) -> float:
            return Q.get((int(z), int(tbin), int(a)), 0.0)

        def best_action(z: int, tbin: int) -> int:
            actions = self.available_actions(
                z,
                spatial_only=spatial_only_actions,
                include_stay=True,
                candidate_nodes=state_nodes,
                max_actions=max_actions_per_state,
            )
            return max(actions, key=lambda a: q_value(z, tbin, a)) if actions else int(z)

        for ep in range(max(1, episodes)):
            frac = ep / max(1, episodes - 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            z = int(origin)
            tbin = 0
            while tbin < horizon_bins:
                actions = self.available_actions(
                    z,
                    spatial_only=spatial_only_actions,
                    include_stay=True,
                    candidate_nodes=state_nodes,
                    max_actions=max_actions_per_state,
                )
                if not actions:
                    break
                if rng.random() < epsilon:
                    a = rng.choice(actions)
                else:
                    a = best_action(z, tbin)
                reward, travel_time, _distance = self._action_reward(
                    z,
                    a,
                    default_step_min=bin_minutes,
                    time_budget_min=time_budget_min,
                    uniform_income=uniform_income,
                )
                step_bins = max(1, int(math.ceil(travel_time / bin_minutes)))
                next_t = min(horizon_bins, tbin + step_bins)
                done = next_t >= horizon_bins
                if done:
                    target = reward
                else:
                    next_actions = self.available_actions(
                        a,
                        spatial_only=spatial_only_actions,
                        include_stay=True,
                        candidate_nodes=state_nodes,
                        max_actions=max_actions_per_state,
                    )
                    next_best = max([q_value(a, next_t, aa) for aa in next_actions], default=0.0)
                    target = reward + gamma * next_best
                old = q_value(z, tbin, a)
                Q[(z, tbin, a)] = old + alpha * (target - old)
                z = int(a)
                tbin = next_t

        records = []
        for z in sorted(state_nodes):
            for tbin in range(horizon_bins):
                actions = self.available_actions(
                    z,
                    spatial_only=spatial_only_actions,
                    include_stay=True,
                    candidate_nodes=state_nodes,
                    max_actions=max_actions_per_state,
                )
                if not actions:
                    continue
                a = max(actions, key=lambda aa: q_value(z, tbin, aa))
                records.append(
                    {
                        "zone": z,
                        "time_bin": tbin,
                        "minutes_elapsed": tbin * bin_minutes,
                        "best_action_zone": int(a),
                        "best_action_name": self.node_info.get(int(a), {}).get("zone_name", ""),
                        "q_value": q_value(z, tbin, a),
                    }
                )
        return pd.DataFrame(records), Q

    def export_dqn_training_records(
        self,
        origin: int,
        time_budget_min: float = DEFAULT_TIME_BUDGET_MIN,
        bin_minutes: float = DEFAULT_TIME_BIN_MIN,
        candidate_nodes: Optional[Sequence[int]] = None,
        spatial_only_actions: bool = True,
        uniform_income: bool = False,
        max_actions_per_state: Optional[int] = 12,
    ) -> pd.DataFrame:
        """Export transition records for a future DQN module.

        This keeps the core environment stable and dependency-light. A future
        PyTorch DQN can train on these (s, a, r, s') records with experience replay
        and a target network, as described in the notes.
        """
        self.validate_zone(origin)
        horizon_bins = int(math.ceil(time_budget_min / bin_minutes))
        if candidate_nodes is None:
            candidate_nodes = self.candidate_opportunity_nodes(origin, time_budget_min, max_candidates=80)
            candidate_nodes = list(set(candidate_nodes) | {int(origin)})
        state_nodes = set(int(z) for z in candidate_nodes if int(z) in self.G.nodes)
        state_nodes.add(int(origin))

        records = []
        for z in sorted(state_nodes):
            for tbin in range(horizon_bins):
                actions = self.available_actions(
                    z,
                    spatial_only=spatial_only_actions,
                    include_stay=True,
                    candidate_nodes=state_nodes,
                    max_actions=max_actions_per_state,
                )
                for a in actions:
                    reward, travel_time, distance = self._action_reward(
                        z,
                        a,
                        default_step_min=bin_minutes,
                        time_budget_min=time_budget_min,
                        uniform_income=uniform_income,
                    )
                    step_bins = max(1, int(math.ceil(travel_time / bin_minutes)))
                    next_t = min(horizon_bins, tbin + step_bins)
                    records.append(
                        {
                            "state_zone": z,
                            "state_time_bin": tbin,
                            "action_zone": int(a),
                            "next_zone": int(a),
                            "next_time_bin": next_t,
                            "reward": reward,
                            "travel_time_min": travel_time,
                            "distance_miles": distance,
                            "done": next_t >= horizon_bins,
                            "state_income_score": float(self.node_info[z].get("income_score", 0.0)),
                            "state_demand_score": float(self.node_info[z].get("demand_score", 0.0)),
                            "action_income_score": 1.0 if uniform_income else float(self.node_info[int(a)].get("income_score", 0.0)),
                            "action_demand_score": float(self.node_info[int(a)].get("demand_score", 0.0)),
                        }
                    )
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Legacy O-D corridor compatibility
    # -------------------------------------------------------------------------

    def legacy_candidate_corridor_nodes(
        self,
        origin: int,
        destination: int,
        hop_radius: int = 1,
        max_candidates: int = 80,
    ) -> List[int]:
        """Old O-D corridor function kept for compatibility."""
        base_path = self.shortest_time_path(origin, destination).path
        candidates = set(base_path)

        spatial_graph = self._spatial_only_graph()
        for node in base_path:
            if node not in spatial_graph:
                continue
            lengths = nx.single_source_shortest_path_length(spatial_graph, int(node), cutoff=hop_radius)
            candidates.update(int(x) for x in lengths.keys())

        if len(candidates) <= max_candidates:
            return sorted(candidates)

        oinfo = self.node_info[int(origin)]
        dinfo = self.node_info[int(destination)]
        line = LineString([(oinfo["centroid_x"], oinfo["centroid_y"]), (dinfo["centroid_x"], dinfo["centroid_y"])])
        scored = []
        for z in candidates:
            info = self.node_info[int(z)]
            dist = line.distance(Point(info["centroid_x"], info["centroid_y"]))
            scored.append((dist, int(z)))
        scored.sort()
        return sorted([z for _, z in scored[:max_candidates]])


# =============================================================================
# Build / load functions
# =============================================================================

def build_environment() -> TaxiOpportunityEnvironment:
    ensure_output_dir()

    zones = load_zone_geometries()
    print(f"Loaded {len(zones)} taxi zone polygons")
    print(f"CRS: {zones.crs}")

    nodes_basic = build_nodes_table(zones)
    valid_zone_ids = set(nodes_basic["zone_id"].astype(int))

    spatial_edges = build_spatial_adjacency_edges(zones)
    print(f"Spatial adjacency directed edges: {len(spatial_edges):,}")

    trips = load_and_clean_all_trips(MONTHS, valid_zone_ids)

    nodes = build_node_income_table(trips, nodes_basic)
    od_weights = build_od_weights(trips)
    print(f"Historical OD edges after filtering: {len(od_weights):,}")

    edges = build_edges(spatial_edges, od_weights)
    print(f"Final directed opportunity edges: {len(edges):,}")

    G = build_networkx_graph(nodes, edges)
    print(f"NetworkX DiGraph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    nodes_path = os.path.join(OUTPUT_DIR, f"opportunity_nodes_{MONTH_TAG}.csv")
    od_path = os.path.join(OUTPUT_DIR, f"opportunity_od_weights_{MONTH_TAG}.csv")
    edges_path = os.path.join(OUTPUT_DIR, f"opportunity_edges_{MONTH_TAG}.csv")
    graph_path = os.path.join(OUTPUT_DIR, f"opportunity_graph_{MONTH_TAG}.pkl")

    nodes.to_csv(nodes_path, index=False)
    od_weights.to_csv(od_path, index=False)
    edges.to_csv(edges_path, index=False)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"Saved nodes: {nodes_path}")
    print(f"Saved OD weights: {od_path}")
    print(f"Saved edges: {edges_path}")
    print(f"Saved graph: {graph_path}")

    return TaxiOpportunityEnvironment(G, nodes, edges)


def load_environment_from_processed() -> TaxiOpportunityEnvironment:
    graph_path = os.path.join(OUTPUT_DIR, f"opportunity_graph_{MONTH_TAG}.pkl")
    nodes_path = os.path.join(OUTPUT_DIR, f"opportunity_nodes_{MONTH_TAG}.csv")
    edges_path = os.path.join(OUTPUT_DIR, f"opportunity_edges_{MONTH_TAG}.csv")
    if not (os.path.exists(graph_path) and os.path.exists(nodes_path) and os.path.exists(edges_path)):
        raise FileNotFoundError("Processed opportunity files not found. Run build_environment() first.")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    return TaxiOpportunityEnvironment(G, nodes, edges)


# =============================================================================
# Demo / quick test
# =============================================================================

def quick_test(env: TaxiOpportunityEnvironment) -> None:
    origin = 132  # JFK Airport
    print("\n=== Quick test: static algorithms + opportunity ranking ===")
    print(f"Origin: {origin} ({env.node_info[origin].get('zone_name')})")

    for alg in ["utility", "dijkstra_time", "dijkstra_distance", "highest_income", "highest_demand"]:
        top = env.rank_opportunity_nodes(origin=origin, time_budget_min=120, top_k=5, algorithm=alg)
        print(f"\nTop nodes by {alg}:")
        cols = [
            "destination",
            "zone_name",
            "borough",
            "travel_time_min",
            "distance_miles",
            "avg_driver_income_per_trip",
            "pickup_rate_per_hour",
            "recommendation_utility",
        ]
        if not top.empty:
            print(top[cols].to_string(index=False))
        else:
            print("No reachable zones found.")

    candidates = env.candidate_opportunity_nodes(origin=origin, time_budget_min=120, max_candidates=40)
    print(f"\nCandidate opportunity nodes for later Shapley: {len(candidates)}")
    print(candidates[:40])

    paths = env.build_candidate_path_pool(
        origin=origin,
        time_budget_min=120,
        candidate_nodes=candidates,
        m_per_family=5,
        final_k=10,
    )
    print("\nTop diverse utility-ranked candidate paths:")
    for i, pr in enumerate(paths, 1):
        print(
            f"{i}. alg={pr.algorithm}, dest={pr.destination}, path={pr.path}, "
            f"time={pr.total_travel_time:.2f} min, dist={pr.total_distance:.2f} miles, "
            f"dest_income={pr.total_expected_income:.2f}, utility={pr.route_utility:.4f}"
        )

    path_table, node_to_bit = env.shapley_ready_path_table(origin, paths)
    shapley_path_path = os.path.join(OUTPUT_DIR, "shapley_ready_candidate_paths.csv")
    path_table.to_csv(shapley_path_path, index=False)
    print(f"\nSaved Shapley-ready path table: {shapley_path_path}")
    print(f"Optional Shapley player nodes: {len(node_to_bit)}")

    # Fast, small RL demonstrations. These are not heavy training tasks.
    print("\n=== Quick test: finite-horizon value iteration on candidate nodes ===")
    vi_df, _V, _policy = env.finite_horizon_value_iteration(
        origin=origin,
        time_budget_min=120,
        candidate_nodes=candidates[:40],
        max_actions_per_state=8,
    )
    vi_path = os.path.join(OUTPUT_DIR, "value_iteration_policy_preview.csv")
    vi_df.to_csv(vi_path, index=False)
    print(f"Saved value iteration policy preview: {vi_path}")
    print(vi_df[vi_df["time_bin"] == 0].head(10).to_string(index=False))

    dqn_records = env.export_dqn_training_records(
        origin=origin,
        time_budget_min=120,
        candidate_nodes=candidates[:40],
        max_actions_per_state=8,
    )
    dqn_path = os.path.join(OUTPUT_DIR, "dqn_transition_records_preview.csv")
    dqn_records.to_csv(dqn_path, index=False)
    print(f"Saved DQN transition-record preview: {dqn_path}")


if __name__ == "__main__":
    environment = build_environment()
    quick_test(environment)
