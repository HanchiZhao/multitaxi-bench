"""Create clear visual reports for the dynamic two-hour taxi project."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DIR, ZONE_SHP, ensure_directories

try:
    import geopandas as gpd
except Exception:
    gpd = None


POLICY_LABELS = {
    "wait_only": "Wait only",
    "highest_demand": "Highest demand",
    "highest_income": "Highest income",
    "greedy_net_earnings": "Greedy net earnings",
    "finite_horizon_value_iteration": "Finite-horizon value iteration",
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "legacy_utility_greedy": "Legacy utility greedy",
}


def load_zones():
    if gpd is None or not ZONE_SHP.exists():
        return None
    zones = gpd.read_file(ZONE_SHP)
    loc_col = next((c for c in zones.columns if c.lower() == "locationid"), None)
    if loc_col is None:
        return None
    zones = zones.rename(columns={loc_col: "zone_id"})
    zones["zone_id"] = pd.to_numeric(zones["zone_id"], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["zone_id", "geometry"]).copy()
    zones["zone_id"] = zones["zone_id"].astype(int)
    return zones


def plot_algorithm_comparison() -> None:
    path = PROCESSED_DIR / "algorithm_recommendation_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("mean_net_earnings_2h")
    labels = [POLICY_LABELS.get(x, x) for x in df["algorithm"]]
    xerr = np.vstack(
        [
            df["mean_net_earnings_2h"] - df["ci_lower"],
            df["ci_upper"] - df["mean_net_earnings_2h"],
        ]
    )
    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.7 * len(df))))
    y = np.arange(len(df))
    bars = ax.barh(y, df["mean_net_earnings_2h"], xerr=xerr, capsize=5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean two-hour net earnings (USD)")
    ax.set_title("Two-Hour Taxi Policy Comparison\nError bars show 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, df["mean_net_earnings_2h"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"  ${value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "algorithm_net_earnings_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_paired_gains() -> None:
    path = PROCESSED_DIR / "paired_policy_comparisons.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("mean_paired_gain")
    if df.empty:
        return
    labels = [POLICY_LABELS.get(x, x) for x in df["algorithm"]]
    y = np.arange(len(df))
    xerr = np.vstack([
        df["mean_paired_gain"] - df["paired_ci_lower"],
        df["paired_ci_upper"] - df["mean_paired_gain"],
    ])
    fig, ax = plt.subplots(figsize=(10, max(5, 0.65 * len(df))))
    bars = ax.barh(y, df["mean_paired_gain"], xerr=xerr, capsize=5)
    ax.set_yticks(y, labels)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Paired net-earnings gain versus WAIT-only (USD / 2h)")
    ax.set_title("Paired Policy Gains on Identical Scenarios\n95% confidence intervals use scenario-level differences")
    ax.grid(axis="x", alpha=0.25)
    for bar, value, win_rate in zip(bars, df["mean_paired_gain"], df["win_rate"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"  ${value:.2f} | wins {win_rate:.0%}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_policy_gains_vs_wait_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_operational_breakdown() -> None:
    path = PROCESSED_DIR / "algorithm_recommendation_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("mean_net_earnings_2h", ascending=False)
    labels = [POLICY_LABELS.get(x, x) for x in df["algorithm"]]
    x = np.arange(len(df))
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 1.5 * len(df)), 10.5))
    axes[0].bar(x - width, df["mean_gross_revenue_2h"], width, label="Gross revenue")
    axes[0].bar(x, df["mean_operating_cost_2h"], width, label="Operating cost")
    axes[0].bar(x + width, df["mean_net_earnings_2h"], width, label="Net earnings")
    axes[0].set_ylabel("USD per two hours")
    axes[0].set_title("Revenue and Cost Breakdown")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    components = [
        ("mean_waiting_minutes", "Waiting"),
        ("mean_empty_minutes", "Empty reposition"),
        ("mean_occupied_minutes", "Completed passenger trips"),
        ("mean_unfinished_occupied_minutes", "Unfinished trip inside horizon"),
        ("mean_terminal_unused_minutes", "Terminal / unused"),
    ]
    bottom = np.zeros(len(df), dtype=float)
    for column, label in components:
        values = df[column].to_numpy(float) if column in df.columns else np.zeros(len(df))
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[1].axhline(120, color="black", linewidth=1.0, linestyle="--", label="120-minute horizon")
    axes[1].set_ylim(0, max(125, float(bottom.max()) + 5))
    axes[1].set_ylabel("Mean minutes")
    axes[1].set_title("Complete Accounting of the Two-Hour Window")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(ncol=2)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "algorithm_operational_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_policy_trajectories(zones) -> None:
    path = PROCESSED_DIR / "policy_trajectories.csv"
    centroid_path = PROCESSED_DIR / "zone_centroids.csv"
    representative_path = PROCESSED_DIR / "representative_trajectories.csv"
    if zones is None or not path.exists() or not centroid_path.exists():
        return
    events = pd.read_csv(path)
    representative = pd.read_csv(representative_path).set_index("algorithm") if representative_path.exists() else pd.DataFrame()
    centroids = pd.read_csv(centroid_path).set_index("zone_id")
    algorithms = events["algorithm"].drop_duplicates().tolist()
    if not algorithms:
        return
    ncols = 2
    nrows = int(math.ceil(len(algorithms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, algorithm in zip(axes, algorithms):
        zones.plot(ax=ax, facecolor="#f1f1f1", edgecolor="#bdbdbd", linewidth=0.35)
        group = events[events["algorithm"] == algorithm].sort_values(["start_elapsed_min", "event_index"])
        movement_number = 0
        for row in group.itertuples(index=False):
            if int(row.from_zone) not in centroids.index or int(row.to_zone) not in centroids.index:
                continue
            x1, y1 = centroids.loc[int(row.from_zone), ["centroid_x", "centroid_y"]]
            x2, y2 = centroids.loc[int(row.to_zone), ["centroid_x", "centroid_y"]]
            if row.event_type == "EMPTY_REPOSITION":
                movement_number += 1
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops={"arrowstyle": "->", "linestyle": "--", "linewidth": 2.0, "color": "#d95f02"})
                ax.scatter([x2], [y2], marker="^", s=34, color="#d95f02", zorder=4)
                ax.text(x2, y2, f"E{movement_number}", fontsize=7, zorder=5)
            elif row.event_type == "OCCUPIED_TRIP":
                movement_number += 1
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops={"arrowstyle": "->", "linewidth": 2.2, "color": "#1b9e77"})
                ax.scatter([x2], [y2], marker="o", s=30, color="#1b9e77", zorder=4)
                ax.text(x2, y2, f"P{movement_number}", fontsize=7, zorder=5)
            elif str(row.event_type).startswith("WAIT"):
                ax.scatter([x1], [y1], marker=".", s=55, color="#202020", zorder=4)
        if not group.empty:
            first_zone = int(group.iloc[0]["from_zone"])
            last_zone = int(group.iloc[-1]["to_zone"])
            if first_zone in centroids.index:
                sx, sy = centroids.loc[first_zone, ["centroid_x", "centroid_y"]]
                ax.scatter([sx], [sy], marker="*", s=180, color="#e41a1c", edgecolor="black", zorder=6)
            if last_zone in centroids.index:
                ex, ey = centroids.loc[last_zone, ["centroid_x", "centroid_y"]]
                ax.scatter([ex], [ey], marker="s", s=60, color="#377eb8", edgecolor="black", zorder=6)
        label = POLICY_LABELS.get(algorithm, algorithm)
        subtitle = ""
        if not representative.empty and algorithm in representative.index:
            record = representative.loc[algorithm]
            subtitle = f"\n{int(record['completed_trips'])} completed trips | net ${float(record['net_earnings_2h']):.2f}"
        ax.set_title(label + subtitle)
        ax.set_axis_off()
    for ax in axes[len(algorithms):]:
        ax.set_visible(False)
    fig.suptitle(
        "Representative Two-Hour Operating Trajectories\n"
        "P = passenger movement; E = empty reposition; star = start; square = end\n"
        "Repeated OD trips can overlap geographically; the companion timeline shows every event",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "algorithm_two_hour_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_timelines() -> None:
    path = PROCESSED_DIR / "policy_trajectories.csv"
    if not path.exists():
        return
    events = pd.read_csv(path)
    algorithms = events["algorithm"].drop_duplicates().tolist()
    if not algorithms:
        return
    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.8 * len(algorithms))))
    y_map = {algorithm: i for i, algorithm in enumerate(algorithms)}
    styles = {
        "WAIT_NO_PICKUP": ("#7f7f7f", "Waiting"),
        "WAIT_FOR_PICKUP": ("#7f7f7f", "Waiting"),
        "WAIT_NO_OD_OPTIONS": ("#7f7f7f", "Waiting"),
        "EMPTY_REPOSITION": ("#d95f02", "Empty reposition"),
        "OCCUPIED_TRIP": ("#1b9e77", "Completed passenger trip"),
        "UNFINISHED_TRIP": ("#7570b3", "Unfinished passenger trip"),
        "TERMINAL_UNUSED": ("#e6ab02", "Terminal / unused"),
        "TERMINAL_OR_INVALID_ROUTE_WAIT": ("#7f7f7f", "Waiting"),
    }
    seen_labels = set()
    for row in events.itertuples(index=False):
        y = y_map[row.algorithm]
        color, label = styles.get(str(row.event_type), ("#cccccc", str(row.event_type)))
        duration = max(0.0, float(row.end_elapsed_min) - float(row.start_elapsed_min))
        plot_label = label if label not in seen_labels else None
        ax.barh(y, duration, left=float(row.start_elapsed_min), height=0.58, color=color, edgecolor="white", label=plot_label)
        seen_labels.add(label)
    ax.set_yticks(range(len(algorithms)), [POLICY_LABELS.get(a, a) for a in algorithms])
    ax.set_xlim(0, 120)
    ax.set_xlabel("Elapsed minutes")
    ax.set_title("Every Event in the Representative Two-Hour Episode")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "algorithm_two_hour_timelines.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trip_count_distribution() -> None:
    path = PROCESSED_DIR / "policy_episode_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    algorithms = df["algorithm"].drop_duplicates().tolist()
    data = [df[df["algorithm"] == algorithm]["completed_trips"].to_numpy(float) for algorithm in algorithms]
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(algorithms)), 6))
    ax.boxplot(data, tick_labels=[POLICY_LABELS.get(a, a) for a in algorithms], showmeans=True)
    ax.set_ylabel("Completed passenger trips within 120 minutes")
    ax.set_title("Trip-Count Distribution Across Common Scenarios")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "algorithm_completed_trip_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shapley(zones) -> None:
    path = PROCESSED_DIR / "node_shapley_values.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("shapley_value", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.65 * len(df))))
    y = np.arange(len(df))
    xerr = np.vstack([df["shapley_value"] - df["ci_lower"], df["ci_upper"] - df["shapley_value"]])
    bars = ax.barh(y, df["shapley_value"], xerr=xerr, capsize=4)
    ax.set_yticks(y, [f"{int(r.zone_id)}  {r.zone_name}" for r in df.itertuples(index=False)])
    ax.set_xlabel("Expected contribution to two-hour net earnings (USD)")
    ax.set_title("Dynamic Zone Shapley Values\nError bars show 95% confidence intervals")
    ax.axvline(0, linewidth=0.8, color="black")
    ax.grid(axis="x", alpha=0.25)
    max_abs = max(1e-12, float(df["shapley_value"].abs().max()))
    for bar, value in zip(bars, df["shapley_value"]):
        if max_abs < 0.10:
            label = f"  ${value:.4f}"
        elif abs(value) < 0.01:
            label = f"  {100*value:.2f}¢"
        else:
            label = f"  ${value:.2f}"
        ax.text(value, bar.get_y() + bar.get_height() / 2, label, va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_dynamic_zone_shapley.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if zones is not None:
        # Only selected Shapley players receive a value. Grey zones were not players and must
        # not be misread as having a mathematically proven zero contribution.
        merged = zones.merge(df[["zone_id", "shapley_value"]], on="zone_id", how="left")
        fig, ax = plt.subplots(figsize=(12, 11))
        tested = merged[merged["shapley_value"].notna()]
        untested = merged[merged["shapley_value"].isna()]
        untested.plot(ax=ax, facecolor="#eeeeee", edgecolor="#b0b0b0", linewidth=0.30)
        if not tested.empty:
            vmax = float(tested["shapley_value"].max())
            vmin = float(tested["shapley_value"].min())
            if abs(vmax-vmin) < 1e-12:
                vmax = vmin + 1e-9
            tested.plot(
                ax=ax, column="shapley_value", cmap="viridis", legend=True,
                edgecolor="#444444", linewidth=0.55, vmin=vmin, vmax=vmax,
                legend_kwds={"label": "Expected 2-hour net-earnings contribution (USD)"},
            )
        ax.set_title(
            "Dynamic Zone Shapley Map\n"
            "Colored zones are evaluated players; grey zones were not included in this Shapley game"
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dynamic_zone_shapley_map.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_ablation() -> None:
    path = PROCESSED_DIR / "ablation_study_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    top = df[df["ablation_type"] == "cumulative_top_shapley"]
    random = df[df["ablation_type"] == "random_control"]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(top["ablation_rank"], top["mean_absolute_drop"], marker="o", linewidth=2.5, label="Remove top-Shapley zones")
    if not random.empty:
        ax.plot(random["ablation_rank"], random["mean_absolute_drop"], marker="s", linewidth=2.0, label="Remove random zones")
        if "std_absolute_drop" in random.columns:
            low = random["mean_absolute_drop"] - random["std_absolute_drop"].fillna(0)
            high = random["mean_absolute_drop"] + random["std_absolute_drop"].fillna(0)
            ax.fill_between(random["ablation_rank"], low, high, alpha=0.18)
    ax.set_xlabel("Number of removed zones")
    ax.set_ylabel("Drop in optimal expected two-hour net earnings (USD)")
    ax.set_title("Dynamic Zone Shapley Ablation Test")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shapley_ablation_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_imputation_validation() -> None:
    path = PROCESSED_DIR / "imputation_holdout_validation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    specs = [
        ("true_revenue", "predicted_revenue_without_same_bin_observation", "Driver revenue (USD)"),
        ("true_duration_min", "predicted_duration_without_same_bin_observation", "Trip duration (minutes)"),
        ("true_distance_miles", "predicted_distance_without_same_bin_observation", "Trip distance (miles)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (truth, pred, title) in zip(axes, specs):
        ax.scatter(df[truth], df[pred], s=12, alpha=0.35)
        lo = min(df[truth].min(), df[pred].min())
        hi = max(df[truth].max(), df[pred].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Reconstructed without same-bin observation")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.suptitle("Hierarchical EB Holdout Validation")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "imputation_holdout_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_q_learning_history() -> None:
    path = PROCESSED_DIR / "q_learning_training_history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["episode"], df["episode_net_reward"], linewidth=0.9, alpha=0.30, label="Recorded episode reward")
    window = max(5, min(25, len(df)//10 if len(df)>=10 else 5))
    smooth = df["episode_net_reward"].rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], smooth, linewidth=2.4, label=f"Rolling mean ({window} records)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode net reward (USD)")
    ax.set_title("Q-Learning Training Progress")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q_learning_training_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_dqn_history() -> None:
    train_path = PROCESSED_DIR / "dqn_training_history.csv"
    validation_path = PROCESSED_DIR / "dqn_validation_history.csv"
    if not train_path.exists():
        return
    train = pd.read_csv(train_path)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(train["episode"], train["episode_net_reward"], linewidth=0.8, alpha=0.25, label="Recorded training episode")
    if len(train):
        window = max(5, min(30, len(train) // 10 if len(train) >= 10 else 5))
        ax.plot(train["episode"], train["episode_net_reward"].rolling(window, min_periods=1).mean(),
                linewidth=2.2, label=f"Training rolling mean ({window} records)")
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        ax.plot(validation["episode"], validation["validation_mean_net_earnings"], marker="o", linewidth=2.0, label="Fixed validation scenarios")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Net reward / earnings (USD)")
    ax.set_title("Double DQN Training and Validation Progress")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dqn_training_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_policy_similarity() -> None:
    matrix_path = PROCESSED_DIR / "policy_action_agreement_matrix.csv"
    summary_path = PROCESSED_DIR / "policy_action_summary.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        labels = [POLICY_LABELS.get(x,x) for x in matrix.index]
        fig, ax = plt.subplots(figsize=(max(7,0.9*len(labels)), max(6,0.8*len(labels))))
        image = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j,i,f"{matrix.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
        ax.set_title("Policy Action Agreement Across Shared States")
        fig.colorbar(image, ax=ax, label="Fraction of states with identical action")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "policy_action_agreement_matrix.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    type_path = PROCESSED_DIR / "policy_action_type_agreement_matrix.csv"
    if type_path.exists():
        matrix = pd.read_csv(type_path, index_col=0)
        labels = [POLICY_LABELS.get(x,x) for x in matrix.index]
        fig, ax = plt.subplots(figsize=(max(7,0.9*len(labels)), max(6,0.8*len(labels))))
        image = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j,i,f"{matrix.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
        ax.set_title("Policy Action-Type Agreement (WAIT vs REPOSITION)")
        fig.colorbar(image, ax=ax, label="Fraction of states with same action type")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "policy_action_type_agreement_matrix.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    if summary_path.exists():
        df = pd.read_csv(summary_path).sort_values("wait_action_rate")
        labels = [POLICY_LABELS.get(x,x) for x in df["algorithm"]]
        fig, ax = plt.subplots(figsize=(10,max(5,0.6*len(df))))
        bars = ax.barh(range(len(df)),df["wait_action_rate"])
        ax.set_yticks(range(len(df)),labels)
        ax.set_xlim(0,1)
        ax.set_xlabel("Fraction of diagnostic states choosing WAIT")
        ax.set_title("Policy WAIT-Action Rate")
        for bar,value in zip(bars,df["wait_action_rate"]):
            ax.text(value,bar.get_y()+bar.get_height()/2,f"  {value:.1%}",va="center")
        ax.grid(axis="x",alpha=.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "policy_wait_action_rates.png",dpi=220,bbox_inches="tight")
        plt.close(fig)


def plot_competition_sensitivity() -> None:
    path = PROCESSED_DIR / "competition_sensitivity_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["low", "medium", "high"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm, group in df.groupby("algorithm", sort=False):
        group = group.assign(_order=group["competition_scenario"].map({v: i for i, v in enumerate(order)})).sort_values("_order")
        means = group["mean_net_earnings_2h"].to_numpy(float)
        lower = means - group["ci_lower"].to_numpy(float)
        upper = group["ci_upper"].to_numpy(float) - means
        ax.errorbar(x[:len(group)], means, yerr=np.vstack([lower, upper]), marker="o", capsize=4,
                    label=POLICY_LABELS.get(algorithm, algorithm))
    ax.set_xticks(x, order)
    ax.set_xlabel("Vacant-taxi competition scenario")
    ax.set_ylabel("Mean two-hour net earnings (USD)")
    ax.set_title("Sensitivity to Latent Vacant-Taxi Competition\nError bars show 95% confidence intervals")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "competition_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_competition_map(zones) -> None:
    node_path = PROCESSED_DIR / "dynamic_node_metrics.csv"
    shapley_path = PROCESSED_DIR / "node_shapley_values.csv"
    if zones is None or not node_path.exists():
        return
    node = pd.read_csv(node_path)
    if shapley_path.exists():
        shapley = pd.read_csv(shapley_path)
        start_time = str(shapley["start_time"].iloc[0])
        hour, minute = map(int, start_time.split(":"))
        target_bin = (hour*60 + minute) // 15
    else:
        target_bin = 32
    snapshot = node[node["time_bin"].astype(int) == int(target_bin)].copy()
    merged = zones.merge(snapshot, on="zone_id", how="left")
    specs = [
        ("posterior_pickups_per_bin_day", "Passenger pickup demand\n(expected pickups per 15-min bin/day)"),
        ("latent_vacant_supply_index", "Latent vacant-taxi supply index\n(relative, not observed fleet count)"),
        ("expected_wait_min", "Expected driver wait (minutes)\nselected competition scenario"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (column, title) in zip(axes, specs):
        merged.plot(
            ax=ax, column=column, cmap="viridis", legend=True,
            edgecolor="#777777", linewidth=0.25,
            missing_kwds={"color":"#eeeeee"},
        )
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"Demand, Latent Competition, and Waiting Environment (time bin {target_bin})", fontsize=15)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(FIGURES_DIR / "demand_supply_wait_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_directories()
    zones = load_zones()
    plot_algorithm_comparison()
    plot_paired_gains()
    plot_operational_breakdown()
    plot_policy_trajectories(zones)
    plot_trajectory_timelines()
    plot_trip_count_distribution()
    plot_shapley(zones)
    plot_ablation()
    plot_imputation_validation()
    plot_q_learning_history()
    plot_dqn_history()
    plot_policy_similarity()
    plot_competition_sensitivity()
    plot_latent_competition_map(zones)
    generated = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    print("Generated figures:")
    for name in generated:
        print(f"  {FIGURES_DIR / name}")
    if zones is None:
        print("GeoPandas/shapefile unavailable: non-map figures were still generated.")


if __name__ == "__main__":
    main()
