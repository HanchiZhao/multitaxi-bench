r"""
Visualize Shapley values and recommendation paths for the revised taxi project.

Recommended usage from project root:
    python scripts/visualize_results.py

Outputs:
    results/figures/top_shapley_nodes.png
    results/figures/shapley_map.png
    results/figures/algorithm_paths_map.png
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

ZONE_SHP = os.path.join(DATA_DIR, "taxi_zones.shp")
ZONE_LOOKUP = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
DEFAULT_SHAPLEY = os.path.join(PROCESSED_DIR, "node_shapley_values.csv")
DEFAULT_COMPARISON = os.path.join(PROCESSED_DIR, "algorithm_recommendation_comparison.csv")
DEFAULT_TOP_BAR = os.path.join(FIGURES_DIR, "top_shapley_nodes.png")
DEFAULT_SHAPLEY_MAP = os.path.join(FIGURES_DIR, "shapley_map.png")
DEFAULT_PATH_MAP = os.path.join(FIGURES_DIR, "algorithm_paths_map.png")


def ensure_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_path(path_value) -> List[int]:
    s = str(path_value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        return [int(float(p.strip())) for p in s.split(",") if p.strip()]
    if "-" in s:
        return [int(float(p.strip())) for p in s.split("-") if p.strip()]
    if "," in s:
        return [int(float(p.strip())) for p in s.split(",") if p.strip()]
    return [int(float(s))]


def load_zones() -> gpd.GeoDataFrame:
    zones = gpd.read_file(ZONE_SHP)
    if "LocationID" not in zones.columns:
        possible = [c for c in zones.columns if c.lower() == "locationid"]
        if possible:
            zones = zones.rename(columns={possible[0]: "LocationID"})
        else:
            raise ValueError("Cannot find LocationID in taxi_zones shapefile.")
    zones["LocationID"] = zones["LocationID"].astype(int)
    if os.path.exists(ZONE_LOOKUP):
        lookup = pd.read_csv(ZONE_LOOKUP)
        lookup["LocationID"] = lookup["LocationID"].astype(int)
        keep = [c for c in ["LocationID", "Borough", "Zone", "service_zone"] if c in lookup.columns]
        zones = zones.merge(lookup[keep], on="LocationID", how="left", suffixes=("", "_lookup"))
    zones = zones[zones.geometry.notna()].copy()
    zones = zones[zones.geometry.is_valid].copy()
    return zones


def save_top_shapley_bar(shapley_path: str, output_path: str, top_n: int = 15) -> None:
    if not os.path.exists(shapley_path):
        print(f"Shapley file not found; skipping bar chart: {shapley_path}")
        return
    df = pd.read_csv(shapley_path)
    if df.empty or "shapley_value" not in df.columns:
        print("Shapley table is empty or missing shapley_value; skipping bar chart.")
        return
    top = df.sort_values("shapley_value", ascending=False).head(top_n).copy()
    labels = []
    for _, row in top.iterrows():
        zone_id = int(row.get("zone_id", -1))
        name = str(row.get("zone_name", ""))
        if not name or name.lower() == "nan":
            labels.append(str(zone_id))
        else:
            labels.append(f"{zone_id} {name}")

    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(labels[::-1], top["shapley_value"].values[::-1])
    plt.xlabel("Node Shapley value")
    plt.ylabel("Taxi zone")
    plt.title(f"Top {len(top)} Shapley Nodes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved top Shapley bar chart: {output_path}")


def save_shapley_map(zones: gpd.GeoDataFrame, shapley_path: str, output_path: str) -> None:
    if not os.path.exists(shapley_path):
        print(f"Shapley file not found; skipping map: {shapley_path}")
        return
    shap = pd.read_csv(shapley_path)
    if shap.empty or "zone_id" not in shap.columns or "shapley_value" not in shap.columns:
        print("Shapley table is empty or missing required columns; skipping map.")
        return
    merged = zones.merge(shap[["zone_id", "shapley_value"]], left_on="LocationID", right_on="zone_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 10))
    merged.plot(column="shapley_value", ax=ax, legend=True, missing_kwds={"color": "lightgrey"}, linewidth=0.1, edgecolor="black")
    ax.set_title("Node Shapley Values by NYC Taxi Zone")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved Shapley map: {output_path}")


def path_to_linestring(path: Sequence[int], centroids: dict) -> LineString | None:
    coords = []
    for z in path:
        if int(z) in centroids:
            p = centroids[int(z)]
            coords.append((p.x, p.y))
    if len(coords) < 2:
        return None
    return LineString(coords)


def save_algorithm_paths_map(zones: gpd.GeoDataFrame, comparison_path: str, output_path: str, max_paths: int = 8) -> None:
    if not os.path.exists(comparison_path):
        print(f"Comparison file not found; skipping path map: {comparison_path}")
        return
    comp = pd.read_csv(comparison_path)
    if comp.empty or "path" not in comp.columns:
        print("Comparison table is empty or missing path column; skipping path map.")
        return

    centroids = dict(zip(zones["LocationID"].astype(int), zones.geometry.centroid))
    rows = comp.head(max_paths).copy()

    fig, ax = plt.subplots(figsize=(10, 10))
    zones.boundary.plot(ax=ax, linewidth=0.2)

    plotted = 0
    for _, row in rows.iterrows():
        path = parse_path(row.get("path", ""))
        line = path_to_linestring(path, centroids)
        if line is None:
            continue
        label = str(row.get("algorithm", f"path_{plotted + 1}"))
        gpd.GeoSeries([line], crs=zones.crs).plot(ax=ax, linewidth=2.0, label=label)
        plotted += 1

    if plotted > 0:
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Algorithm Recommendation Paths")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved algorithm paths map: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Shapley values and recommendation paths.")
    parser.add_argument("--shapley", type=str, default=DEFAULT_SHAPLEY)
    parser.add_argument("--comparison", type=str, default=DEFAULT_COMPARISON)
    parser.add_argument("--top-bar-output", type=str, default=DEFAULT_TOP_BAR)
    parser.add_argument("--shapley-map-output", type=str, default=DEFAULT_SHAPLEY_MAP)
    parser.add_argument("--path-map-output", type=str, default=DEFAULT_PATH_MAP)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--max-paths", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    save_top_shapley_bar(args.shapley, args.top_bar_output, top_n=args.top_n)

    try:
        zones = load_zones()
    except Exception as exc:
        print(f"Could not load taxi zone shapefile; skipping maps. Reason: {exc}")
        return

    save_shapley_map(zones, args.shapley, args.shapley_map_output)
    save_algorithm_paths_map(zones, args.comparison, args.path_map_output, max_paths=args.max_paths)

    print("\n=== Visualization complete ===")
    print(f"Top Shapley bar: {args.top_bar_output}")
    print(f"Shapley map: {args.shapley_map_output}")
    print(f"Algorithm path map: {args.path_map_output}")


if __name__ == "__main__":
    main()
