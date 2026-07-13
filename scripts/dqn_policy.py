"""Dynamic state-action DQN policy.

The network scores one (state, action) pair at a time, so every taxi state may expose a
newly generated action set. This avoids a fixed output head tied to a frozen zone list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from config import (
    DQN_DYNAMIC_CANDIDATES, EMPTY_COST_PER_MILE, HORIZON_MINUTES,
    MAX_EXPECTED_WAIT_MINUTES, MAX_REPOSITION_MILES, MAX_REPOSITION_MINUTES,
    MODELS_DIR, OCCUPIED_COST_PER_MILE,
)
from two_hour_environment import DynamicTaxiEnvironment, WAIT_ACTION, parse_clock_time


def state_features(env: DynamicTaxiEnvironment, zone: int, elapsed: float, start_time: str) -> np.ndarray:
    absolute = parse_clock_time(start_time) + float(elapsed)
    metric = env.node_metric(int(zone), absolute)
    angle = 2.0 * np.pi * ((absolute % 1440.0) / 1440.0)
    remaining = max(0.0, HORIZON_MINUTES - float(elapsed)) / HORIZON_MINUTES
    return np.array([
        np.sin(angle), np.cos(angle), remaining,
        metric["pickup_rate_index"] / max(1.0, float(env.metadata.get("pickup_scale", 10.0))),
        metric["expected_wait_min"] / MAX_EXPECTED_WAIT_MINUTES,
        metric["expected_trip_revenue"] / 100.0,
        metric["expected_trip_duration_min"] / 90.0,
        metric["expected_trip_distance_miles"] / 30.0,
        metric["competition_ratio"] / (1.0 + metric["competition_ratio"]),
        metric.get("estimated_competitors_equivalent", 0.0) /
        (10.0 + metric.get("estimated_competitors_equivalent", 0.0)),
        metric["confidence"],
    ], dtype=np.float32)


def state_action_features(
    env: DynamicTaxiEnvironment,
    zone: int,
    elapsed: float,
    start_time: str,
    action: str | int,
) -> np.ndarray:
    absolute = parse_clock_time(start_time) + float(elapsed)
    base = state_features(env, zone, elapsed, start_time)
    if action == WAIT_ACTION:
        target = env.node_metric(int(zone), absolute)
        extra = np.array([
            1.0, 0.0, 0.0, 0.0,
            target["pickup_rate_index"] / max(1.0, float(env.metadata.get("pickup_scale", 10.0))),
            target["expected_wait_min"] / MAX_EXPECTED_WAIT_MINUTES,
            (target["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE * target["expected_trip_distance_miles"]) / 100.0,
            target["expected_trip_duration_min"] / 90.0,
            target["confidence"],
            0.0,
        ], dtype=np.float32)
    else:
        target_zone = int(action)
        route = env.route(int(zone), target_zone, absolute_minutes=absolute)
        arrival = absolute + float(route["duration_min"])
        target = env.node_metric(target_zone, arrival)
        extra = np.array([
            0.0,
            float(route["duration_min"]) / MAX_REPOSITION_MINUTES,
            float(route["distance_miles"]) / MAX_REPOSITION_MILES,
            EMPTY_COST_PER_MILE * float(route["distance_miles"]) / 20.0,
            target["pickup_rate_index"] / max(1.0, float(env.metadata.get("pickup_scale", 10.0))),
            target["expected_wait_min"] / MAX_EXPECTED_WAIT_MINUTES,
            (target["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE * target["expected_trip_distance_miles"]) / 100.0,
            target["expected_trip_duration_min"] / 90.0,
            target["confidence"],
            float(route["confidence"]),
        ], dtype=np.float32)
    return np.concatenate([base, extra]).astype(np.float32)


def build_network(input_dim: int, hidden_dims: tuple[int, ...] = (192, 128, 64)):
    import torch.nn as nn

    layers: list[nn.Module] = []
    last = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(last, int(width)), nn.LayerNorm(int(width)), nn.SiLU()])
        last = int(width)
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


@dataclass
class DQNPolicy:
    model: object
    start_time: str
    candidate_count: int = DQN_DYNAMIC_CANDIDATES
    device: str = "cpu"
    name: str = "dqn"
    decision_count: int = 0

    def reset_diagnostics(self) -> None:
        self.decision_count = 0

    def diagnostics(self) -> Dict[str, float]:
        return {
            "algorithm": self.name,
            "decision_count": int(self.decision_count),
            "model_action_count": int(self.decision_count),
            "model_action_rate": 1.0 if self.decision_count else 0.0,
            "fallback_action_count": 0,
            "fallback_action_rate": 0.0,
            "unseen_state_count": 0,
            "unseen_state_rate": 0.0,
        }

    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        import torch
        # Small state-action batches are much faster with one CPU thread; large default
        # thread pools otherwise dominate inference time.
        if self.device == "cpu" and torch.get_num_threads() != 1:
            torch.set_num_threads(1)

        self.decision_count += 1
        actions: list[str | int] = [WAIT_ACTION] + [
            int(z) for z in allowed_zones if int(z) != int(state.zone_id)
        ]
        features = np.stack([
            state_action_features(env, state.zone_id, state.elapsed_minutes, self.start_time, action)
            for action in actions
        ])
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(tensor).squeeze(-1).cpu().numpy()
        return actions[int(np.argmax(q_values))]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "DQNPolicy":
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        path = Path(path or (MODELS_DIR / "dqn_policy.pt"))
        payload = torch.load(path, map_location="cpu")
        input_dim = int(payload["input_dim"])
        hidden_dims = tuple(int(x) for x in payload.get("hidden_dims", (192, 128, 64)))
        model = build_network(input_dim, hidden_dims)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(
            model=model,
            start_time=str(payload["start_time"]),
            candidate_count=int(payload.get("candidate_count", DQN_DYNAMIC_CANDIDATES)),
        )
