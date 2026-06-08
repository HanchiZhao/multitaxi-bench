"""
Route Environment for TLC Taxi Zone Path Modeling.

This script builds a route-level NYC taxi zone environment.

Core idea:
- Each TLC taxi zone is a graph node.
- Spatial adjacency and historical taxi OD trips define graph edges.
- Edge weights are estimated from TLC Yellow Taxi trip records.
- The environment supports route search between a given origin and destination.

This file only covers environment modeling.
It does NOT compute Shapley values yet.
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx


# ============================================================
# Configuration
# ============================================================

# Current file:
# multitaxi-bench/scripts/route_environment.py
# Project root:
# multitaxi-bench/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MONTHS = [
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
]


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
FARE_MIN = 2.5
FARE_MAX = 500.0
DIST_MIN = 0.01
DIST_MAX = 100.0
DURATION_MIN = 1.0
DURATION_MAX = 240.0

DRIVER_SHARE = 0.70

# EPSG:2263 uses feet.
ADJACENCY_DISTANCE_THRESHOLD_FEET = 200.0

# If historical OD edge is too rare, ignore it as a noisy edge.
MIN_OD_TRIP_COUNT = 5

# Fallback speed for spatial edges without historical OD data.
FALLBACK_SPEED_MPH = 12.0

# Utility weights for route recommendation.
DEFAULT_INCOME_WEIGHT = 1.0
DEFAULT_DEMAND_WEIGHT = 0.1
DEFAULT_TIME_WEIGHT = 1.0


# ============================================================
# Utility functions
# ============================================================

def safe_read_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Please run: pip install pyarrow"
        ) from e


def ensure_columns(df: pd.DataFrame, required_cols: List[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def feet_to_miles(feet: float) -> float:
    return float(feet) / 5280.0


def miles_to_minutes(distance_miles: float, speed_mph: float = FALLBACK_SPEED_MPH) -> float:
    if speed_mph <= 0:
        raise ValueError("speed_mph must be positive.")
    return float(distance_miles) / speed_mph * 60.0


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    min_v = s.min()
    max_v = s.max()

    if max_v - min_v <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - min_v) / (max_v - min_v)


# ============================================================
# Zone loading and node construction
# ============================================================

def load_zone_geometries(
    shapefile_path: str,
    lookup_path: str,
) -> gpd.GeoDataFrame:
    print("[1/6] Loading taxi zone geometries...")

    zones = gpd.read_file(shapefile_path)
    lookup = pd.read_csv(lookup_path)

    ensure_columns(
        zones,
        ["LocationID", "borough", "zone", "geometry"],
        "taxi_zones.shp",
    )

    ensure_columns(
        lookup,
        ["LocationID", "Borough", "Zone", "service_zone"],
        "taxi_zone_lookup.csv",
    )

    zones["LocationID"] = zones["LocationID"].astype(int)
    lookup["LocationID"] = lookup["LocationID"].astype(int)

    lookup = lookup.rename(
        columns={
            "Borough": "lookup_borough",
            "Zone": "lookup_zone",
        }
    )

    zones = zones.merge(
        lookup[["LocationID", "lookup_borough", "lookup_zone", "service_zone"]],
        on="LocationID",
        how="left",
    )

    zones["borough"] = zones["lookup_borough"].fillna(zones["borough"])
    zones["zone"] = zones["lookup_zone"].fillna(zones["zone"])

    zones = zones.drop(columns=["lookup_borough", "lookup_zone"])

    zones["geometry"] = zones["geometry"].buffer(0)

    if zones.crs is None:
        print("  Warning: CRS is missing. Assuming EPSG:2263.")
        zones = zones.set_crs(epsg=2263)

    if zones.crs.to_epsg() != 2263:
        print(f"  Reprojecting from {zones.crs} to EPSG:2263.")
        zones = zones.to_crs(epsg=2263)

    zones["centroid"] = zones.geometry.centroid
    zones["centroid_x"] = zones["centroid"].x
    zones["centroid_y"] = zones["centroid"].y
    zones["area"] = zones.geometry.area

    zones = zones.sort_values("LocationID").reset_index(drop=True)

    print(f"  Loaded {len(zones)} taxi zone polygons.")
    print(f"  CRS: {zones.crs}")

    return zones


def build_nodes_table(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    nodes = zones[
        [
            "LocationID",
            "borough",
            "zone",
            "service_zone",
            "centroid_x",
            "centroid_y",
            "area",
        ]
    ].copy()

    nodes = nodes.rename(
        columns={
            "LocationID": "zone_id",
            "zone": "zone_name",
        }
    )

    nodes_path = os.path.join(OUTPUT_DIR, "route_nodes.csv")
    nodes.to_csv(nodes_path, index=False)
    print(f"  Saved nodes: {nodes_path}")

    return nodes


# ============================================================
# Spatial adjacency construction
# ============================================================

def build_spatial_adjacency_edges(
    zones: gpd.GeoDataFrame,
    distance_threshold_feet: float = ADJACENCY_DISTANCE_THRESHOLD_FEET,
) -> pd.DataFrame:
    """
    Build spatial adjacency edges from zone polygons.

    Two zones are connected if:
    - their polygons touch/intersect, or
    - their polygon distance is below a small threshold.

    The output is directed: if i is adjacent to j, both i->j and j->i are included.
    """
    print("[2/6] Building spatial adjacency edges...")

    centroids = dict(zip(zones["LocationID"].astype(int), zones.geometry.centroid))
    edges = []

    sindex = zones.sindex

    for idx, row in zones.iterrows():
        src_id = int(row["LocationID"])
        geom = row.geometry

        buffered_bounds = geom.buffer(distance_threshold_feet).bounds
        candidate_idx = list(sindex.intersection(buffered_bounds))

        for j in candidate_idx:
            if j == idx:
                continue

            dst_id = int(zones.iloc[j]["LocationID"])
            dst_geom = zones.iloc[j].geometry

            touches = geom.touches(dst_geom)
            intersects = geom.intersects(dst_geom)
            distance = geom.distance(dst_geom)

            if touches or intersects or distance <= distance_threshold_feet:
                centroid_distance_feet = centroids[src_id].distance(centroids[dst_id])
                centroid_distance_miles = feet_to_miles(centroid_distance_feet)
                fallback_travel_time = miles_to_minutes(centroid_distance_miles)

                edges.append(
                    {
                        "from_zone": src_id,
                        "to_zone": dst_id,
                        "edge_source": "spatial_adjacency",
                        "polygon_distance_feet": float(distance),
                        "centroid_distance_miles": centroid_distance_miles,
                        "fallback_travel_time_min": fallback_travel_time,
                    }
                )

    adjacency_edges = pd.DataFrame(edges).drop_duplicates(
        subset=["from_zone", "to_zone"]
    )

    adjacency_path = os.path.join(OUTPUT_DIR, "spatial_adjacency_edges.csv")
    adjacency_edges.to_csv(adjacency_path, index=False)

    print(f"  Spatial adjacency directed edges: {len(adjacency_edges)}")
    print(f"  Saved spatial edges: {adjacency_path}")

    return adjacency_edges


# ============================================================
# TLC trip data OD aggregation
# ============================================================

def clean_trip_data(df: pd.DataFrame, valid_zone_ids: set, month: str) -> pd.DataFrame:
    ensure_columns(df, TRIP_USE_COLS, f"yellow_tripdata_{month}.parquet")

    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce",
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"],
        errors="coerce",
    )

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

    df = df[
        df["PULocationID"].isin(valid_zone_ids)
        & df["DOLocationID"].isin(valid_zone_ids)
    ]

    duration_min = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    df["duration_min"] = duration_min

    df["tip_amount"] = pd.to_numeric(df["tip_amount"], errors="coerce").fillna(0.0)
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(
        df["fare_amount"] + df["tip_amount"]
    )
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")

    df = df[
        (df["fare_amount"] >= FARE_MIN)
        & (df["fare_amount"] <= FARE_MAX)
        & (df["trip_distance"] >= DIST_MIN)
        & (df["trip_distance"] <= DIST_MAX)
        & (df["duration_min"] >= DURATION_MIN)
        & (df["duration_min"] <= DURATION_MAX)
    ]

    df["driver_income"] = DRIVER_SHARE * (
        df["fare_amount"] + df["tip_amount"].fillna(0.0)
    )

    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["minute"] = df["tpep_pickup_datetime"].dt.minute
    df["time_bin"] = df["hour"] * 4 + df["minute"] // 15

    return df


def build_od_weights(
    months: List[str],
    valid_zone_ids: set,
) -> pd.DataFrame:
    print("[3/6] Building OD weights from yellow taxi trips...")

    monthly_frames = []

    for month in months:
        path = os.path.join(DATA_DIR, f"yellow_tripdata_{month}.parquet")

        if not os.path.exists(path):
            print(f"  Warning: missing {path}, skipping.")
            continue

        print(f"  Reading {month}: {path}")
        raw = safe_read_parquet(path, columns=TRIP_USE_COLS)
        raw_count = len(raw)

        clean = clean_trip_data(raw, valid_zone_ids, month)
        clean_count = len(clean)

        print(f"    Raw trips: {raw_count:,} -> clean trips: {clean_count:,}")

        monthly_frames.append(clean)

    if not monthly_frames:
        raise FileNotFoundError(
            "No valid yellow_tripdata parquet files found. "
            "Please place yellow_tripdata_YYYY-MM.parquet in data/."
        )

    trips = pd.concat(monthly_frames, ignore_index=True)

    od = (
        trips.groupby(["PULocationID", "DOLocationID"])
        .agg(
            trip_count=("PULocationID", "size"),
            avg_distance_miles=("trip_distance", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_tip=("tip_amount", "mean"),
            avg_total_amount=("total_amount", "mean"),
            avg_driver_income=("driver_income", "mean"),
        )
        .reset_index()
    )

    od = od.rename(
        columns={
            "PULocationID": "from_zone",
            "DOLocationID": "to_zone",
        }
    )

    od = od[od["trip_count"] >= MIN_OD_TRIP_COUNT].copy()

    pickup_demand = (
        trips.groupby("PULocationID")
        .size()
        .reset_index(name="pickup_demand")
        .rename(columns={"PULocationID": "zone_id"})
    )

    dropoff_demand = (
        trips.groupby("DOLocationID")
        .size()
        .reset_index(name="dropoff_demand")
        .rename(columns={"DOLocationID": "zone_id"})
    )

    node_demand = pickup_demand.merge(dropoff_demand, on="zone_id", how="outer").fillna(0)
    node_demand["total_node_demand"] = (
        node_demand["pickup_demand"] + node_demand["dropoff_demand"]
    )

    node_demand_path = os.path.join(OUTPUT_DIR, "node_demand_2025-01_2025-06.csv")
    node_demand.to_csv(node_demand_path, index=False)

    od_path = os.path.join(OUTPUT_DIR, "od_weights_2025-01_2025-06.csv")
    od.to_csv(od_path, index=False)

    print(f"  OD edges after filtering: {len(od)}")
    print(f"  Saved OD weights: {od_path}")
    print(f"  Saved node demand: {node_demand_path}")

    return od


# ============================================================
# Edge merge and graph construction
# ============================================================

def build_route_edges(
    adjacency_edges: pd.DataFrame,
    od_weights: pd.DataFrame,
) -> pd.DataFrame:
    print("[4/6] Combining spatial and historical OD edges...")

    adjacency = adjacency_edges.copy()
    od = od_weights.copy()

    edges = adjacency.merge(
        od,
        on=["from_zone", "to_zone"],
        how="left",
    )

    edges["has_historical_od"] = edges["trip_count"].notna()
    edges["trip_count"] = edges["trip_count"].fillna(0).astype(int)

    edges["avg_distance_miles"] = edges["avg_distance_miles"].fillna(
        edges["centroid_distance_miles"]
    )

    edges["avg_duration_min"] = edges["avg_duration_min"].fillna(
        edges["fallback_travel_time_min"]
    )

    edges["avg_driver_income"] = edges["avg_driver_income"].fillna(0.0)
    edges["avg_fare"] = edges["avg_fare"].fillna(0.0)
    edges["avg_tip"] = edges["avg_tip"].fillna(0.0)
    edges["avg_total_amount"] = edges["avg_total_amount"].fillna(0.0)

    existing_pairs = set(zip(edges["from_zone"], edges["to_zone"]))

    od_extra = od[
        ~od.apply(
            lambda r: (int(r["from_zone"]), int(r["to_zone"])) in existing_pairs,
            axis=1,
        )
    ].copy()

    if len(od_extra) > 0:
        od_extra["edge_source"] = "historical_od"
        od_extra["polygon_distance_feet"] = np.nan
        od_extra["centroid_distance_miles"] = od_extra["avg_distance_miles"]
        od_extra["fallback_travel_time_min"] = od_extra["avg_duration_min"]
        od_extra["has_historical_od"] = True

        od_extra = od_extra[
            [
                "from_zone",
                "to_zone",
                "edge_source",
                "polygon_distance_feet",
                "centroid_distance_miles",
                "fallback_travel_time_min",
                "trip_count",
                "avg_distance_miles",
                "avg_duration_min",
                "avg_fare",
                "avg_tip",
                "avg_total_amount",
                "avg_driver_income",
                "has_historical_od",
            ]
        ]

        edges = pd.concat([edges, od_extra], ignore_index=True)

    edges["income_score"] = normalize_series(edges["avg_driver_income"])
    edges["demand_score"] = normalize_series(edges["trip_count"])
    edges["time_score"] = normalize_series(edges["avg_duration_min"])

    edges["edge_utility"] = (
        DEFAULT_INCOME_WEIGHT * edges["income_score"]
        + DEFAULT_DEMAND_WEIGHT * edges["demand_score"]
        - DEFAULT_TIME_WEIGHT * edges["time_score"]
    )

    edges["travel_time_cost"] = edges["avg_duration_min"]

    # For max-utility path, smaller cost is better.
    edges["utility_cost"] = -edges["edge_utility"]

    edges = edges.drop_duplicates(subset=["from_zone", "to_zone"]).reset_index(drop=True)

    edges_path = os.path.join(OUTPUT_DIR, "route_edges.csv")
    edges.to_csv(edges_path, index=False)

    print(f"  Final directed route edges: {len(edges)}")
    print(f"  Saved route edges: {edges_path}")

    return edges


def build_networkx_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> nx.DiGraph:
    print("[5/6] Building NetworkX DiGraph...")

    G = nx.DiGraph()

    for _, row in nodes.iterrows():
        zone_id = int(row["zone_id"])
        G.add_node(
            zone_id,
            zone_id=zone_id,
            zone_name=row.get("zone_name", ""),
            borough=row.get("borough", ""),
            service_zone=row.get("service_zone", ""),
            centroid_x=float(row["centroid_x"]),
            centroid_y=float(row["centroid_y"]),
            area=float(row["area"]),
        )

    for _, row in edges.iterrows():
        u = int(row["from_zone"])
        v = int(row["to_zone"])

        if u not in G.nodes or v not in G.nodes:
            continue

        G.add_edge(
            u,
            v,
            edge_source=row["edge_source"],
            has_historical_od=bool(row["has_historical_od"]),
            trip_count=int(row["trip_count"]),
            avg_distance_miles=float(row["avg_distance_miles"]),
            avg_duration_min=float(row["avg_duration_min"]),
            avg_fare=float(row["avg_fare"]),
            avg_tip=float(row["avg_tip"]),
            avg_total_amount=float(row["avg_total_amount"]),
            avg_driver_income=float(row["avg_driver_income"]),
            income_score=float(row["income_score"]),
            demand_score=float(row["demand_score"]),
            time_score=float(row["time_score"]),
            edge_utility=float(row["edge_utility"]),
            travel_time_cost=float(row["travel_time_cost"]),
            utility_cost=float(row["utility_cost"]),
        )

    graph_path = os.path.join(OUTPUT_DIR, "nyc_taxi_zone_graph.pkl")

    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Saved graph: {graph_path}")

    return G


# ============================================================
# Route Environment Class
# ============================================================

@dataclass
class PathResult:
    origin: int
    destination: int
    path: List[int]
    total_travel_time: float
    total_distance: float
    total_expected_income: float
    total_demand_score: float
    total_utility: float
    num_edges: int


class TaxiRouteEnvironment:
    """
    Route-level environment over NYC TLC taxi zones.
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        graph: Optional[nx.DiGraph] = None,
    ):
        if graph is not None:
            self.G = graph
        else:
            if graph_path is None:
                graph_path = os.path.join(OUTPUT_DIR, "nyc_taxi_zone_graph.pkl")

            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)

        self.zone_ids = list(self.G.nodes())

    def validate_zone(self, zone_id: int) -> None:
        if zone_id not in self.G.nodes:
            raise ValueError(f"Zone {zone_id} is not in the graph.")

    def edge_data(self, u: int, v: int) -> Dict[str, Any]:
        return self.G.edges[u, v]

    def compute_path_result(self, path: List[int]) -> PathResult:
        if len(path) < 2:
            raise ValueError("Path must contain at least origin and destination.")

        origin = int(path[0])
        destination = int(path[-1])

        total_travel_time = 0.0
        total_distance = 0.0
        total_expected_income = 0.0
        total_demand_score = 0.0
        total_utility = 0.0

        for u, v in zip(path[:-1], path[1:]):
            if not self.G.has_edge(u, v):
                raise ValueError(f"Invalid path: missing edge {u} -> {v}")

            data = self.edge_data(u, v)

            total_travel_time += data["avg_duration_min"]
            total_distance += data["avg_distance_miles"]
            total_expected_income += data["avg_driver_income"]
            total_demand_score += data["demand_score"]
            total_utility += data["edge_utility"]

        return PathResult(
            origin=origin,
            destination=destination,
            path=[int(x) for x in path],
            total_travel_time=float(total_travel_time),
            total_distance=float(total_distance),
            total_expected_income=float(total_expected_income),
            total_demand_score=float(total_demand_score),
            total_utility=float(total_utility),
            num_edges=len(path) - 1,
        )

    def shortest_time_path(self, origin: int, destination: int) -> PathResult:
        self.validate_zone(origin)
        self.validate_zone(destination)

        path = nx.shortest_path(
            self.G,
            source=origin,
            target=destination,
            weight="travel_time_cost",
        )

        return self.compute_path_result(path)

    def highest_utility_path(self, origin: int, destination: int) -> PathResult:
        """
        Diagnostic only.

        Because utility_cost can be negative, this can be unstable if negative
        cycles exist. Later, route recommendation should use k-shortest path
        ranking instead of direct global utility shortest path.
        """
        self.validate_zone(origin)
        self.validate_zone(destination)

        path = nx.shortest_path(
            self.G,
            source=origin,
            target=destination,
            weight="utility_cost",
        )

        return self.compute_path_result(path)

    def k_shortest_time_paths(
        self,
        origin: int,
        destination: int,
        k: int = 10,
    ) -> List[PathResult]:
        self.validate_zone(origin)
        self.validate_zone(destination)

        generator = nx.shortest_simple_paths(
            self.G,
            source=origin,
            target=destination,
            weight="travel_time_cost",
        )

        results = []
        for idx, path in enumerate(generator):
            if idx >= k:
                break
            results.append(self.compute_path_result(path))

        return results

    def candidate_corridor_nodes(
        self,
        origin: int,
        destination: int,
        hop_radius: int = 1,
        max_candidates: int = 80,
    ) -> List[int]:
        """
        Generate candidate nodes for O-D Shapley analysis.

        Important:
        The full graph contains many historical OD edges, so it is very dense.
        If we use k-hop expansion on the full graph, almost all NYC zones will
        be selected. Therefore, this function uses spatial adjacency edges only
        to build the local corridor.

        Strategy:
        1. Find shortest-time path in the full weighted graph.
        2. Build a spatial-only graph using edge_source == "spatial_adjacency".
        3. Add nodes within hop_radius from each shortest-path node.
        4. If candidate set is still too large, keep zones closest to the
           origin-destination straight-line corridor.
        """
        self.validate_zone(origin)
        self.validate_zone(destination)

        base_path = self.shortest_time_path(origin, destination).path
        candidate = set(base_path)

        spatial_graph = nx.Graph()

        for u, v, data in self.G.edges(data=True):
            if data.get("edge_source") == "spatial_adjacency":
                spatial_graph.add_edge(u, v)

        for node in base_path:
            if node not in spatial_graph:
                continue

            lengths = nx.single_source_shortest_path_length(
                spatial_graph,
                node,
                cutoff=hop_radius,
            )
            candidate.update(lengths.keys())

        candidate = list(candidate)

        if len(candidate) <= max_candidates:
            return sorted(int(x) for x in candidate)

        ox = self.G.nodes[origin]["centroid_x"]
        oy = self.G.nodes[origin]["centroid_y"]
        dx = self.G.nodes[destination]["centroid_x"]
        dy = self.G.nodes[destination]["centroid_y"]

        def point_line_distance(px, py, ax, ay, bx, by):
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay

            denom = abx * abx + aby * aby

            if denom <= 1e-12:
                return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)

            t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
            closest_x = ax + t * abx
            closest_y = ay + t * aby

            return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

        scored = []

        for z in candidate:
            zx = self.G.nodes[z]["centroid_x"]
            zy = self.G.nodes[z]["centroid_y"]

            dist = point_line_distance(
                zx,
                zy,
                ox,
                oy,
                dx,
                dy,
            )

            scored.append((z, dist))

        scored = sorted(scored, key=lambda x: x[1])

        base_set = set(base_path)
        selected = list(base_set)

        for z, _ in scored:
            if z not in selected:
                selected.append(z)

            if len(selected) >= max_candidates:
                break

        return sorted(int(x) for x in selected)


# ============================================================
# Main build pipeline
# ============================================================

def build_environment() -> TaxiRouteEnvironment:
    print("=" * 80)
    print("Building NYC TLC Taxi Zone Route Environment")
    print("=" * 80)

    shapefile_path = os.path.join(DATA_DIR, "taxi_zones.shp")
    lookup_path = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")

    zones = load_zone_geometries(
        shapefile_path=shapefile_path,
        lookup_path=lookup_path,
    )

    nodes = build_nodes_table(zones)

    valid_zone_ids = set(nodes["zone_id"].astype(int).tolist())

    adjacency_edges = build_spatial_adjacency_edges(zones)

    od_weights = build_od_weights(
        months=MONTHS,
        valid_zone_ids=valid_zone_ids,
    )

    route_edges = build_route_edges(
        adjacency_edges=adjacency_edges,
        od_weights=od_weights,
    )

    G = build_networkx_graph(
        nodes=nodes,
        edges=route_edges,
    )

    env = TaxiRouteEnvironment(graph=G)

    print("[6/6] Environment build complete.")
    print("=" * 80)

    return env


def quick_test(env: TaxiRouteEnvironment) -> None:
    """
    Run a small route test.

    Example:
    132 = JFK Airport
    161 = Midtown Center
    """
    print("\nQuick route test:")

    origin = 132
    destination = 161

    if origin not in env.G.nodes or destination not in env.G.nodes:
        print(f"  Test zones {origin}->{destination} not found in graph.")
        return

    try:
        shortest = env.shortest_time_path(origin, destination)

        print(f"  Shortest-time path {origin}->{destination}:")
        print(f"    Path: {shortest.path}")
        print(f"    Travel time: {shortest.total_travel_time:.2f} min")
        print(f"    Distance: {shortest.total_distance:.2f} miles")
        print(f"    Expected income: {shortest.total_expected_income:.2f}")
        print(f"    Utility: {shortest.total_utility:.4f}")

        candidates = env.candidate_corridor_nodes(
            origin=origin,
            destination=destination,
            hop_radius=1,
            max_candidates=80,
        )

        print(f"  Candidate corridor nodes: {len(candidates)}")
        print(f"    {candidates[:40]}{'...' if len(candidates) > 40 else ''}")

    except nx.NetworkXNoPath:
        print(f"  No path found from {origin} to {destination}.")


if __name__ == "__main__":
    environment = build_environment()
    quick_test(environment)