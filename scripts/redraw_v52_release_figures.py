"""Redraw four publication figures from the accepted v5.2 artifact tables.

This release-only helper changes layout and annotation placement, never numerical data.
It is intentionally separate from the protected scientific pipeline.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "paper_artifacts" / "final_20260811"
POLICY_LABELS = {
    "wait_only": "Wait only",
    "highest_demand": "Highest demand",
    "highest_income": "Highest income",
    "greedy_net_earnings": "Greedy net earnings",
    "finite_horizon_value_iteration": "Finite-horizon value iteration",
    "q_learning": "Q-learning",
    "dqn": "DQN",
}
PATH_LABELS = {
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
POSITIVE_COLOR = "#c94841"
NEGATIVE_COLOR = "#3478b8"
NEUTRAL_COLOR = "#9aa1a8"


def short_text(value: object, width: int = 20) -> str:
    text = " ".join(str(value).split())
    return textwrap.shorten(text, width=width, placeholder="…")


def informative_occurrences(
    group: pd.DataFrame, max_positive: int = 5, max_negative: int = 3
) -> pd.DataFrame:
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


def policy_comparison(tables: Path, figures: Path) -> None:
    df = pd.read_csv(tables / "algorithm_recommendation_comparison.csv").sort_values(
        "mean_net_earnings_2h"
    )
    labels = [POLICY_LABELS.get(value, value) for value in df["algorithm"]]
    xerr = np.vstack(
        [
            df["mean_net_earnings_2h"] - df["ci_lower"],
            df["ci_upper"] - df["mean_net_earnings_2h"],
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.7 * len(df))))
    y = np.arange(len(df))
    bars = ax.barh(y, df["mean_net_earnings_2h"], xerr=xerr, capsize=5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean two-hour net earnings (USD)")
    ax.set_title("Two-Hour Taxi Policy Comparison\nError bars show 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    label_pad = max(0.35, 0.012 * float(df["ci_upper"].max()))
    for bar, value, ci_upper in zip(
        bars, df["mean_net_earnings_2h"], df["ci_upper"]
    ):
        ax.text(
            float(ci_upper) + label_pad,
            bar.get_y() + bar.get_height() / 2,
            f"${value:.2f}",
            va="center",
            ha="left",
        )
    ax.set_xlim(0.0, float(df["ci_upper"].max()) + 5.8)
    fig.tight_layout()
    fig.savefig(figures / "algorithm_net_earnings_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def paired_gains(tables: Path, figures: Path) -> None:
    df = pd.read_csv(tables / "paired_policy_comparisons.csv").sort_values(
        "mean_paired_gain"
    )
    labels = [POLICY_LABELS.get(value, value) for value in df["algorithm"]]
    y = np.arange(len(df))
    xerr = np.vstack(
        [
            df["mean_paired_gain"] - df["paired_ci_lower"],
            df["paired_ci_upper"] - df["mean_paired_gain"],
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, max(5, 0.65 * len(df))))
    bars = ax.barh(y, df["mean_paired_gain"], xerr=xerr, capsize=5)
    ax.set_yticks(y, labels)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Paired net-earnings gain versus WAIT-only (USD / 2h)")
    ax.set_title(
        "Paired Policy Gains on Identical Scenarios\n"
        "95% confidence intervals use scenario-level differences"
    )
    ax.grid(axis="x", alpha=0.25)
    span = float(df["paired_ci_upper"].max() - df["paired_ci_lower"].min())
    label_pad = max(0.18, 0.012 * span)
    for bar, value, ci_lower, ci_upper, win_rate in zip(
        bars,
        df["mean_paired_gain"],
        df["paired_ci_lower"],
        df["paired_ci_upper"],
        df["win_rate"],
    ):
        positive = float(value) >= 0.0
        ax.text(
            float(ci_upper) + label_pad if positive else float(ci_lower) - label_pad,
            bar.get_y() + bar.get_height() / 2,
            f"${value:+.2f} | wins {win_rate:.0%}",
            va="center",
            ha="left" if positive else "right",
        )
    ax.set_xlim(
        float(df["paired_ci_lower"].min()) - 4.0,
        float(df["paired_ci_upper"].max()) + 4.0,
    )
    fig.tight_layout()
    fig.savefig(figures / "paired_policy_gains_vs_wait_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def competition_sensitivity(tables: Path, figures: Path) -> None:
    df = pd.read_csv(tables / "competition_sensitivity_results.csv")
    order = ["low", "medium", "high"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    order_map = {value: index for index, value in enumerate(order)}
    for algorithm, group in df.groupby("algorithm", sort=False):
        group = group.assign(
            _order=group["competition_scenario"].map(order_map)
        ).sort_values("_order")
        means = group["mean_net_earnings_2h"].to_numpy(float)
        lower = means - group["ci_lower"].to_numpy(float)
        upper = group["ci_upper"].to_numpy(float) - means
        ax.errorbar(
            x[: len(group)],
            means,
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
            label=POLICY_LABELS.get(algorithm, algorithm),
        )
    ax.set_xticks(x, order)
    ax.set_xlabel("Vacant-taxi competition scenario")
    ax.set_ylabel("Mean two-hour net earnings (USD)")
    ax.set_title(
        "Sensitivity to Latent Vacant-Taxi Competition\n"
        "Error bars show 95% confidence intervals"
    )
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(figures / "competition_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def path_shapley_bars(tables: Path, figures: Path) -> None:
    values = pd.read_csv(tables / "path_shapley_values.csv")
    algorithms = list(values["algorithm"].drop_duplicates())
    columns = min(4, max(1, len(algorithms)))
    rows = max(1, math.ceil(len(algorithms) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.7 * columns, 5.2 * rows), squeeze=False
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
        ax.set_title(PATH_LABELS.get(algorithm, algorithm), loc="left", fontweight="bold")
        ax.set_xlabel("Contribution to two-hour operating net earnings (USD)")
        ax.tick_params(axis="y", labelsize=7.5)
        for bar, value in zip(bars, group["shapley_value"]):
            value = float(value)
            offset = 0.025 * local_max
            ax.text(
                value + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7.8,
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
            fontsize=7.4,
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
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965), h_pad=2.0, w_pad=1.5)
    fig.savefig(figures / "algorithm_path_shapley_bars.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    tables = artifact_root / "tables"
    figures = artifact_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    policy_comparison(tables, figures)
    paired_gains(tables, figures)
    competition_sensitivity(tables, figures)
    path_shapley_bars(tables, figures)
    print("V5.2 release figures redrawn from accepted artifact tables:")
    for name in (
        "algorithm_net_earnings_comparison.png",
        "paired_policy_gains_vs_wait_only.png",
        "competition_sensitivity.png",
        "algorithm_path_shapley_bars.png",
    ):
        print(figures / name)


if __name__ == "__main__":
    main()
