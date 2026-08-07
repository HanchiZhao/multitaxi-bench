"""V5.1 cost and algorithm-conditioned path-occurrence visualizations.

The path and Shapley definitions are not changed here. The bar figure selects the most
informative signed occurrences for legibility, while the map still draws every occurrence
over the official NYC Taxi Zone polygons.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import textwrap

import geopandas as gpd
from matplotlib import colors as mcolors
from matplotlib import lines as mlines
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DIR, ZONE_SHP, ensure_directories
from cost_models import load_yaml, primary_cost_model


POSITIVE_COLOR = "#c94841"
NEGATIVE_COLOR = "#3478b8"
NEUTRAL_COLOR = "#9aa1a8"
PATH_COLOR = "#263746"
BACKGROUND_COLOR = "#eef1f3"
ZONE_EDGE_COLOR = "#ffffff"
ALGORITHM_LABELS = {
    "wait_only": "Wait Only",
    "highest_demand": "Highest Demand",
    "highest_income": "Highest Income",
    "greedy_net_earnings": "Greedy Net Earnings",
    "finite_horizon_value_iteration": "Finite-Horizon Value Iteration",
    "q_learning": "Q-Learning",
    "dqn": "DQN",
}
ROLE_LABELS = {
    "passenger_dropoff": "passenger drop-off",
    "algorithm_reposition_target": "reposition target",
    "waiting_location": "waiting location",
    "other_event_endpoint": "other event",
    "other": "other event",
}
ROLE_MARKERS = {
    "passenger_dropoff": "o",
    "algorithm_reposition_target": "D",
    "waiting_location": "s",
    "other_event_endpoint": "s",
    "other": "s",
}


def display_algorithm(value: str) -> str:
    value = str(value)
    return ALGORITHM_LABELS.get(value, value.replace("_", " ").title())


def short_text(value: object, width: int = 24) -> str:
    text = " ".join(str(value).split())
    return textwrap.shorten(text, width=width, placeholder="…")


def informative_occurrences(
    group: pd.DataFrame, max_positive: int = 5, max_negative: int = 3
) -> pd.DataFrame:
    """Select signed extremes without modifying or aggregating Shapley values."""
    positive = group[group["shapley_value"] > 1e-12].nlargest(
        max_positive, "shapley_value"
    )
    negative = group[group["shapley_value"] < -1e-12].nsmallest(
        max_negative, "shapley_value"
    )
    selected = pd.concat([positive, negative]).loc[
        lambda frame: ~frame.index.duplicated()
    ]
    capacity = max_positive + max_negative
    if len(selected) < min(capacity, len(group)):
        remaining = group.drop(index=selected.index, errors="ignore").copy()
        remaining["abs_shapley"] = remaining["shapley_value"].abs()
        selected = pd.concat(
            [selected, remaining.nlargest(capacity - len(selected), "abs_shapley")]
        )
    return selected.sort_values("shapley_value", ascending=True).copy()


def cost_figures(summary: pd.DataFrame, model, episodes: pd.DataFrame) -> None:
    value_col = (
        "mean_primary_operating_net_earnings_2h"
        if "mean_primary_operating_net_earnings_2h" in summary
        else "mean_primary_take_home_2h"
    )
    ordered = summary.sort_values(value_col, ascending=True)
    lower = ordered[value_col] - ordered["ci_lower"]
    upper = ordered["ci_upper"] - ordered[value_col]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(
        [display_algorithm(value) for value in ordered["algorithm"]],
        ordered[value_col],
        xerr=np.vstack([lower, upper]),
        color="#4e7fa8",
        edgecolor="white",
        linewidth=0.6,
        capsize=4,
    )
    ax.axvline(0.0, color="#2f3438", linewidth=1.0)
    ax.set_xlabel("Mean two-hour operating net earnings (USD)")
    ax.set_title(
        "Policy Performance After Explicit Cost Accounting\n"
        f"Primary cost model: {model.name}"
    )
    ax.grid(axis="x", color="#d7dce0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ordered[value_col]):
        ax.text(
            float(value) + max(0.5, 0.01 * float(ordered[value_col].abs().max())),
            bar.get_y() + bar.get_height() / 2,
            f"${float(value):.1f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "policy_operating_net_earnings_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)

    canonical_suffix = "_operating_net_earnings_2h"
    legacy_suffix = "_take_home_2h"
    model_columns = [
        column
        for column in episodes.columns
        if column.endswith(canonical_suffix)
        and column != "primary_operating_net_earnings_2h"
    ]
    suffix = canonical_suffix
    if not model_columns:
        suffix = legacy_suffix
        model_columns = [
            column
            for column in episodes.columns
            if column.endswith(legacy_suffix) and column != "primary_take_home_2h"
        ]
    rows = []
    for column in model_columns:
        cost_model = column[: -len(suffix)]
        for algorithm, group in episodes.groupby("algorithm"):
            rows.append(
                {
                    "cost_model": cost_model,
                    "algorithm": algorithm,
                    "mean": float(group[column].mean()),
                }
            )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return
    pivot = comparison.pivot(
        index="algorithm", columns="cost_model", values="mean"
    )
    pivot.index = [display_algorithm(value) for value in pivot.index]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("Mean two-hour operating net earnings (USD)")
    ax.set_xlabel("")
    ax.set_title("Cost-Model Sensitivity")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", color="#d7dce0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Cost model", frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cost_model_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def path_bar_figure(values: pd.DataFrame, algorithms: list[str]) -> None:
    columns = 2
    rows = max(1, math.ceil(len(algorithms) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.5, 5.0 * rows),
        squeeze=False,
    )
    for ax, algorithm in zip(axes.flat, algorithms):
        full_group = values[values["algorithm"] == algorithm].sort_values(
            "path_position"
        )
        group = informative_occurrences(full_group)
        labels = [
            f"P{int(row.path_position):02d} · Z{int(row.zone_id)} · "
            f"{short_text(row.zone_name)}\n"
            f"{ROLE_LABELS.get(str(row.node_role), 'other event')}"
            for row in group.itertuples(index=False)
        ]
        colors = [
            POSITIVE_COLOR
            if value > 1e-12
            else NEGATIVE_COLOR
            if value < -1e-12
            else NEUTRAL_COLOR
            for value in group["shapley_value"]
        ]
        bars = ax.barh(
            np.arange(len(group)),
            group["shapley_value"],
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            height=0.68,
        )
        ax.set_yticks(np.arange(len(group)), labels)
        ax.axvline(0.0, color="#252a2e", linewidth=1.2, zorder=0)
        local_max = max(1.0, float(group["shapley_value"].abs().max()))
        ax.set_xlim(-1.23 * local_max, 1.23 * local_max)
        ax.grid(axis="x", color="#d7dce0", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_title(display_algorithm(algorithm), loc="left", fontweight="bold")
        ax.set_xlabel("Contribution to two-hour operating net earnings (USD)")
        ax.tick_params(axis="y", labelsize=8.5)
        for bar, value in zip(bars, group["shapley_value"]):
            value = float(value)
            offset = 0.025 * local_max
            ax.text(
                value + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.5,
                color="#222222",
            )
        hidden = len(full_group) - len(group)
        ax.text(
            0.99,
            0.02,
            f"{len(group)} shown / {len(full_group)} occurrences"
            + (f" ({hidden} smaller omitted)" if hidden else ""),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#626970",
        )

    for ax in axes.flat[len(algorithms) :]:
        ax.axis("off")
    fig.suptitle(
        "Algorithm-Conditioned Path-Occurrence Shapley Contributions",
        fontsize=16,
        y=0.995,
    )
    fig.text(
        0.5,
        0.004,
        "Left of zero = negative signed contribution; right of zero = positive. "
        "Each panel shows up to five strongest positive and three strongest negative "
        "occurrences; no Shapley value is clipped or changed.",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.98), h_pad=2.4, w_pad=2.0)
    fig.savefig(
        FIGURES_DIR / "algorithm_path_shapley_bars.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def zone_geometries() -> tuple[gpd.GeoDataFrame, dict[int, object]]:
    if not Path(ZONE_SHP).exists():
        raise FileNotFoundError(
            f"Taxi Zone polygon file not found: {ZONE_SHP}. The path map will not "
            "fall back to a blank longitude/latitude plot."
        )
    zones = gpd.read_file(ZONE_SHP)
    if zones.crs is None:
        raise ValueError(f"Taxi Zone shapefile has no CRS: {ZONE_SHP}")
    if "LocationID" not in zones.columns:
        raise ValueError("Taxi Zone shapefile must contain LocationID")
    zones = zones[["LocationID", "geometry"]].copy()
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones = zones.dissolve(by="LocationID", as_index=False)
    if not bool(zones.geometry.is_valid.all()):
        zones["geometry"] = zones.geometry.buffer(0)
    points = {
        int(row.LocationID): row.geometry.representative_point()
        for row in zones.itertuples(index=False)
    }
    return zones, points


def point_xy(point) -> tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def draw_arrow(ax, start, end, *, linewidth: float = 1.6, scale: float = 10) -> None:
    if start is None or end is None:
        return
    x0, y0 = point_xy(start)
    x1, y1 = point_xy(end)
    if math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        return
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=linewidth,
        color=PATH_COLOR,
        alpha=0.75,
        shrinkA=4,
        shrinkB=4,
        zorder=3,
    )
    ax.add_patch(arrow)


def expanded_extent(
    bounds: tuple[float, float, float, float],
    *,
    minimum_span: float = 7000.0,
    padding_fraction: float = 0.14,
    target_aspect: float = 1.18,
) -> tuple[float, float, float, float]:
    """Pad a local Taxi Zone bound while avoiding an unusably tight one-zone crop."""
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    center_x = 0.5 * (minx + maxx)
    center_y = 0.5 * (miny + maxy)
    width = max(maxx - minx, minimum_span)
    height = max(maxy - miny, minimum_span)
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    xpad = padding_fraction * width
    ypad = padding_fraction * height
    return (
        center_x - 0.5 * width - xpad,
        center_y - 0.5 * height - ypad,
        center_x + 0.5 * width + xpad,
        center_y + 0.5 * height + ypad,
    )


def focus_extent(
    zones: gpd.GeoDataFrame, group: pd.DataFrame, start_zone: int
) -> tuple[float, float, float, float]:
    """Focus the main panel on path-occurrence zones, excluding a distant start."""
    occurrence_zones = set(group["zone_id"].astype(int))
    local_zones = occurrence_zones - {int(start_zone)}
    if not local_zones:
        local_zones = occurrence_zones or {int(start_zone)}
    selected = zones[zones["LocationID"].astype(int).isin(local_zones)]
    if selected.empty:
        raise ValueError(f"No Taxi Zone polygons found for local zones: {sorted(local_zones)}")
    return expanded_extent(tuple(selected.total_bounds))


def point_inside_extent(
    point, extent: tuple[float, float, float, float]
) -> bool:
    x, y = point_xy(point)
    minx, miny, maxx, maxy = extent
    return minx <= x <= maxx and miny <= y <= maxy


def clip_segment_to_extent(
    start, end, extent: tuple[float, float, float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clip so an off-panel JFK leg enters at the detail-map edge."""
    x0, y0 = point_xy(start)
    x1, y1 = point_xy(end)
    minx, miny, maxx, maxy = extent
    dx = x1 - x0
    dy = y1 - y0
    lower = 0.0
    upper = 1.0
    for direction, distance in (
        (-dx, x0 - minx),
        (dx, maxx - x0),
        (-dy, y0 - miny),
        (dy, maxy - y0),
    ):
        if abs(direction) <= 1e-12:
            if distance < 0.0:
                return None
            continue
        ratio = distance / direction
        if direction < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return (
        (x0 + lower * dx, y0 + lower * dy),
        (x0 + upper * dx, y0 + upper * dy),
    )


def occurrence_coordinates(
    group: pd.DataFrame,
    points: dict[int, object],
    extent: tuple[float, float, float, float],
) -> dict[int, tuple[float, float]]:
    """Spread same-zone repeats around their true point for display only."""
    width = extent[2] - extent[0]
    height = extent[3] - extent[1]
    radius = min(1000.0, max(260.0, 0.035 * min(width, height)))
    coordinates: dict[int, tuple[float, float]] = {}
    ordered = group.sort_values("path_position")
    for zone_id, repeated in ordered.groupby("zone_id", sort=False):
        center_x, center_y = point_xy(points[int(zone_id)])
        positions = [int(value) for value in repeated["path_position"]]
        count = len(positions)
        if count == 1:
            coordinates[positions[0]] = (center_x, center_y)
            continue
        for index, position in enumerate(positions):
            angle = (2.0 * math.pi * index / count) + math.pi / 4.0
            coordinates[position] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
    return coordinates


def draw_locator_inset(
    ax,
    zones: gpd.GeoDataFrame,
    points: dict[int, object],
    group: pd.DataFrame,
    start_zone: int,
    detail_extent: tuple[float, float, float, float],
    norm,
    cmap,
) -> None:
    """Keep the full JFK-to-detail geography without shrinking the main panel."""
    inset = ax.inset_axes([0.715, 0.035, 0.265, 0.245], zorder=20)
    zones.plot(
        ax=inset,
        facecolor="#f4f6f7",
        edgecolor="#ffffff",
        linewidth=0.16,
        zorder=0,
    )
    sequence = [points[int(start_zone)]] + [
        points[int(zone)] for zone in group.sort_values("path_position")["zone_id"]
    ]
    for start, end in zip(sequence[:-1], sequence[1:]):
        draw_arrow(inset, start, end, linewidth=0.62, scale=5.2)
    for row in group.itertuples(index=False):
        point = points[int(row.zone_id)]
        if int(row.zone_id) == int(start_zone):
            continue
        inset.scatter(
            [point.x],
            [point.y],
            c=[float(row.shapley_value)],
            cmap=cmap,
            norm=norm,
            s=14,
            marker=ROLE_MARKERS.get(
                str(row.node_role), ROLE_MARKERS["other_event_endpoint"]
            ),
            edgecolors="#20262b",
            linewidths=0.25,
            zorder=5,
        )
    start = points[int(start_zone)]
    inset.scatter(
        [start.x],
        [start.y],
        marker="*",
        s=60,
        facecolor="#f2c14e",
        edgecolor="#20262b",
        linewidth=0.6,
        zorder=7,
    )
    start_rows = group[group["zone_id"].astype(int).eq(int(start_zone))]
    for ring_index, row in enumerate(start_rows.itertuples(index=False)):
        ring_color = cmap(norm(float(row.shapley_value)))
        inset.scatter(
            [start.x],
            [start.y],
            marker="o",
            s=82 + 24 * ring_index,
            facecolors="none",
            edgecolors=[ring_color],
            linewidths=1.2,
            zorder=8,
        )
    minx, miny, maxx, maxy = detail_extent
    inset.add_patch(
        Rectangle(
            (minx, miny),
            maxx - minx,
            maxy - miny,
            fill=False,
            edgecolor="#b07a00",
            linewidth=1.1,
            zorder=8,
        )
    )
    full_minx, full_miny, full_maxx, full_maxy = zones.total_bounds
    full_xpad = 0.02 * (full_maxx - full_minx)
    full_ypad = 0.02 * (full_maxy - full_miny)
    inset.set_xlim(full_minx - full_xpad, full_maxx + full_xpad)
    inset.set_ylim(full_miny - full_ypad, full_maxy + full_ypad)
    inset.set_aspect("equal")
    inset.set_axis_off()
    inset.set_title(
        f"NYC locator · start Z{int(start_zone)}",
        fontsize=6.4,
        fontweight="bold",
        pad=1.5,
    )


def key_indices(group: pd.DataFrame) -> set[int]:
    # Maps have less annotation space than bars. Plot every occurrence, but label only
    # the three strongest positive, two strongest negative, and all reposition targets.
    selected = informative_occurrences(group, max_positive=3, max_negative=2)
    keys = set(int(value) for value in selected["path_position"])
    reposition = group[
        group["node_role"].astype(str).eq("algorithm_reposition_target")
    ]
    keys.update(int(value) for value in reposition["path_position"])
    return keys


def path_map_figure(values: pd.DataFrame, algorithms: list[str]) -> None:
    zones, points = zone_geometries()
    missing = sorted(set(values["zone_id"].astype(int)) - set(points))
    start_missing = sorted(set(values["from_zone"].astype(int)) - set(points))
    if missing:
        raise ValueError(f"Path destination zones missing from Taxi Zone polygons: {missing}")
    if start_missing:
        # from_zone includes intermediate origins, so require complete geometry as well.
        raise ValueError(f"Path origin zones missing from Taxi Zone polygons: {start_missing}")

    all_values = values["shapley_value"].astype(float)
    negative_limit = min(-1e-9, float(all_values.min()))
    positive_limit = max(1e-9, float(all_values.max()))
    maximum = max(abs(negative_limit), abs(positive_limit))
    # The signed sides use their actual extrema. This keeps modest negative values blue
    # instead of nearly white when one large positive trip dominates the shared scale.
    norm = mcolors.TwoSlopeNorm(
        vmin=negative_limit,
        vcenter=0.0,
        vmax=positive_limit,
    )
    cmap = plt.get_cmap("RdBu_r")
    columns = 2
    rows = max(1, math.ceil(len(algorithms) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.5, 5.8 * rows),
        squeeze=False,
    )
    label_offsets = [
        (8, 8),
        (-27, 9),
        (9, -15),
        (-29, -15),
        (15, 2),
        (-34, 2),
        (3, 17),
        (3, -22),
    ]

    for ax, algorithm in zip(axes.flat, algorithms):
        group = values[values["algorithm"] == algorithm].sort_values(
            "path_position"
        )
        if group.empty:
            ax.axis("off")
            continue
        start_zone = int(group.iloc[0]["from_zone"])
        extent = focus_extent(zones, group, start_zone)
        display_points = occurrence_coordinates(group, points, extent)
        zones.plot(
            ax=ax,
            facecolor=BACKGROUND_COLOR,
            edgecolor=ZONE_EDGE_COLOR,
            linewidth=0.34,
            zorder=0,
        )
        sequence = [point_xy(points[start_zone])] + [
            display_points[int(position)] for position in group["path_position"]
        ]
        for start, end in zip(sequence[:-1], sequence[1:]):
            clipped = clip_segment_to_extent(start, end, extent)
            if clipped is not None:
                draw_arrow(ax, clipped[0], clipped[1])

        keys = key_indices(group)
        for row in group.itertuples(index=False):
            point = points[int(row.zone_id)]
            display_x, display_y = display_points[int(row.path_position)]
            value = float(row.shapley_value)
            role = str(row.node_role)
            marker = ROLE_MARKERS.get(role, ROLE_MARKERS["other_event_endpoint"])
            scaled = math.sqrt(abs(value) / maximum) if maximum > 0 else 0.0
            size = 58.0 + 235.0 * scaled
            if math.hypot(display_x - point.x, display_y - point.y) > 1e-9:
                ax.plot(
                    [point.x, display_x],
                    [point.y, display_y],
                    color="#6f7880",
                    linewidth=0.52,
                    alpha=0.7,
                    zorder=4,
                )
            ax.scatter(
                [display_x],
                [display_y],
                c=[value],
                cmap=cmap,
                norm=norm,
                s=size,
                marker=marker,
                edgecolors="#20262b",
                linewidths=0.75,
                zorder=5,
            )
            if (
                int(row.path_position) in keys
                and point_inside_extent((display_x, display_y), extent)
            ):
                offset_distance = math.hypot(
                    display_x - point.x, display_y - point.y
                )
                if offset_distance > 1e-9:
                    offset = (
                        int(round(19.0 * (display_x - point.x) / offset_distance)),
                        int(round(19.0 * (display_y - point.y) / offset_distance)),
                    )
                else:
                    offset = label_offsets[
                        int(row.path_position) % len(label_offsets)
                    ]
                ax.annotate(
                    f"P{int(row.path_position):02d}",
                    (display_x, display_y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#1d2328",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                    annotation_clip=True,
                    zorder=7,
                )

        start = points[start_zone]
        if point_inside_extent(start, extent):
            ax.scatter(
                [start.x],
                [start.y],
                marker="*",
                s=285,
                facecolor="#f2c14e",
                edgecolor="#20262b",
                linewidth=1.0,
                zorder=8,
            )
            ax.annotate(
                f"START · Z{start_zone}",
                (start.x, start.y),
                xytext=(8, -10),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#fff7d6",
                    "edgecolor": "#bca34a",
                    "linewidth": 0.5,
                    "alpha": 0.95,
                },
                zorder=9,
            )
        else:
            draw_locator_inset(
                ax,
                zones,
                points,
                group,
                start_zone,
                extent,
                norm,
                cmap,
            )
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"{display_algorithm(algorithm)}\n"
            f"{len(group)} path occurrences · signed Shapley · local detail",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )

    for ax in axes.flat[len(algorithms) :]:
        ax.axis("off")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_axis = fig.add_axes((0.925, 0.35, 0.015, 0.30))
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="vertical")
    negative_ticks = np.linspace(negative_limit, 0.0, 4)[:-1]
    positive_ticks = np.linspace(0.0, positive_limit, 5)
    colorbar.set_ticks(np.concatenate([negative_ticks, positive_ticks]))
    colorbar.set_label(
        "Path-occurrence Shapley contribution to operating net earnings (USD)\n"
        "negative ← 0 → positive · actual signed limits",
        fontsize=9,
    )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markerfacecolor="#f2c14e",
            markeredgecolor="#20262b",
            markersize=13,
            label="Start zone",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#20262b",
            markersize=8,
            label="Passenger drop-off",
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#20262b",
            markersize=7,
            label="Reposition target",
        ),
        mlines.Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#20262b",
            markersize=7,
            label="Waiting / other endpoint",
        ),
        mlines.Line2D(
            [],
            [],
            color=PATH_COLOR,
            linewidth=1.8,
            label="Trajectory order",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.47, 0.038),
        fontsize=9,
    )
    fig.suptitle(
        "Algorithm-Conditioned Representative-Trajectory Shapley Maps",
        fontsize=16,
        y=0.997,
    )
    fig.text(
        0.47,
        0.003,
        "Main panels are cropped to each algorithm's occurrence region; locator insets "
        "retain the full NYC start-to-path context. Repeated same-zone occurrences are "
        "slightly offset and linked to the zone point for legibility only.\n"
        "Straight arrows connect zone representative points and do not claim street-level "
        "routing. Every occurrence and signed Shapley value remains unchanged; P-labels "
        "mark the most informative occurrences and all reposition targets.",
        ha="center",
        va="bottom",
        fontsize=8.35,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.90,
        top=0.945,
        bottom=0.065,
        hspace=0.15,
        wspace=0.04,
    )
    fig.savefig(
        FIGURES_DIR / "algorithm_conditioned_path_shapley_maps.png",
        dpi=260,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def heatmap(values: pd.DataFrame) -> None:
    # This remains a descriptive cross-context view. Occurrence identity is preserved
    # in path_shapley_values.csv and is never overwritten by this display-only mean.
    matrix = values.pivot_table(
        index=["zone_id", "zone_name"],
        columns="algorithm",
        values="shapley_value",
        aggfunc="mean",
    )
    if matrix.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(9, len(matrix.columns) * 1.35), max(5.5, len(matrix) * 0.35))
    )
    maximum = max(1e-9, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy()),
        aspect="auto",
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
    )
    ax.set_xticks(
        range(len(matrix.columns)),
        [display_algorithm(value) for value in matrix.columns],
        rotation=32,
        ha="right",
    )
    ax.set_yticks(
        range(len(matrix.index)),
        [f"Z{zone} · {short_text(name, 30)}" for zone, name in matrix.index],
    )
    ax.set_title(
        "The Same Taxi Zone Can Have Different Path-Conditioned Shapley Values"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Mean signed occurrence Shapley contribution (USD)",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cross_algorithm_zone_shapley_heatmap.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--only",
        choices=("all", "path-map", "path-bars"),
        default="all",
        help="Redraw all extension figures or only a path figure.",
    )
    args = parser.parse_args()
    ensure_directories()
    config = load_yaml(args.config)
    if args.only == "all":
        model = primary_cost_model(config)
        summary = pd.read_csv(PROCESSED_DIR / "algorithm_cost_model_comparison.csv")
        episodes = pd.read_csv(PROCESSED_DIR / "policy_episode_cost_models.csv")
        cost_figures(summary, model, episodes)

    values = pd.read_csv(PROCESSED_DIR / "path_shapley_values.csv")
    required = {
        "algorithm",
        "path_position",
        "zone_id",
        "zone_name",
        "node_role",
        "from_zone",
        "shapley_value",
    }
    missing = sorted(required - set(values.columns))
    if missing:
        raise SystemExit(f"path_shapley_values.csv is missing columns: {missing}")
    algorithms = list(values["algorithm"].drop_duplicates())
    if args.only in {"all", "path-bars"}:
        path_bar_figure(values, algorithms)
    if args.only == "all":
        heatmap(values)
    if args.only in {"all", "path-map"}:
        path_map_figure(values, algorithms)
    print(f"V5.1 figure selection '{args.only}' saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
