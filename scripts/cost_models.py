"""Explicit V5.1 operating-cost definitions and accounting identities.

The preserved V4 simulator remains the operational core.  A primary cost model is
propagated into V4 through environment variables so every policy, learner, DP value,
and Shapley game uses the same active objective.  Post-hoc re-accounting first recovers
raw modeled receipts and then applies each alternative model exactly once.

For a fare-share model, ``revenue_share`` is authoritative. ``fare_share_rate`` is the
complementary disclosure field; it is reported but never subtracted again.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict

import yaml


ACCOUNTING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CostModel:
    name: str
    revenue_share: float
    occupied_cost_per_mile: float
    empty_cost_per_mile: float
    fixed_lease_per_hour: float = 0.0
    fare_share_rate: float = 0.0
    source_farebox_share_rate: float = 0.0
    tips_retained_fully: bool = False

    def validate(self) -> None:
        numeric = {
            "revenue_share": self.revenue_share,
            "occupied_cost_per_mile": self.occupied_cost_per_mile,
            "empty_cost_per_mile": self.empty_cost_per_mile,
            "fixed_lease_per_hour": self.fixed_lease_per_hour,
            "fare_share_rate": self.fare_share_rate,
            "source_farebox_share_rate": self.source_farebox_share_rate,
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError(f"All cost parameters must be finite: {numeric}")
        if not 0.0 < self.revenue_share <= 1.0:
            raise ValueError(
                f"revenue_share must be in (0, 1], got {self.revenue_share}"
            )
        for label in (
            "occupied_cost_per_mile",
            "empty_cost_per_mile",
            "fixed_lease_per_hour",
            "fare_share_rate",
            "source_farebox_share_rate",
        ):
            value = float(getattr(self, label))
            if value < 0.0:
                raise ValueError(f"{label} must be non-negative")
        if self.fare_share_rate >= 1.0:
            raise ValueError("fare_share_rate must be less than 1")
        if self.source_farebox_share_rate >= 1.0:
            raise ValueError("source_farebox_share_rate must be less than 1")

        if self.fare_share_rate > 0.0:
            identity_gap = abs(
                self.revenue_share + self.fare_share_rate - 1.0
            )
            if identity_gap > ACCOUNTING_TOLERANCE:
                raise ValueError(
                    "Fare-share accounting requires revenue_share + "
                    f"fare_share_rate = 1; gap={identity_gap:.3e}"
                )
            if self.fixed_lease_per_hour > ACCOUNTING_TOLERANCE:
                raise ValueError(
                    "A fare-share lease and a fixed hourly lease cannot be charged "
                    "in the same cost model"
                )

        if self.tips_retained_fully and self.source_farebox_share_rate <= 0.0:
            raise ValueError(
                "tips_retained_fully requires a positive source_farebox_share_rate"
            )
        if (
            self.source_farebox_share_rate > 0.0
            and self.fare_share_rate - self.source_farebox_share_rate
            > ACCOUNTING_TOLERANCE
        ):
            raise ValueError(
                "The effective withheld share of fare+tip receipts cannot exceed "
                "the source farebox-only share when tips are retained"
            )

    def env(self) -> Dict[str, str]:
        self.validate()
        return {
            "MULTITAXI_COST_MODEL": self.name,
            "MULTITAXI_DRIVER_REVENUE_SHARE": str(self.revenue_share),
            "MULTITAXI_OCCUPIED_COST_PER_MILE": str(
                self.occupied_cost_per_mile
            ),
            "MULTITAXI_EMPTY_COST_PER_MILE": str(self.empty_cost_per_mile),
            "MULTITAXI_FIXED_LEASE_PER_HOUR": str(self.fixed_lease_per_hour),
            "MULTITAXI_FARE_SHARE_RATE": str(self.fare_share_rate),
        }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def all_cost_models(config: Dict[str, Any]) -> Dict[str, CostModel]:
    raw = config.get("cost_models", {})
    models: Dict[str, CostModel] = {}
    for name, values in raw.items():
        if name == "primary" or not isinstance(values, dict):
            continue
        models[name] = CostModel(
            name=name,
            revenue_share=float(values.get("revenue_share", 1.0)),
            occupied_cost_per_mile=float(
                values.get("occupied_cost_per_mile", 0.35)
            ),
            empty_cost_per_mile=float(
                values.get("empty_cost_per_mile", 0.35)
            ),
            fixed_lease_per_hour=float(
                values.get("fixed_lease_per_hour", 0.0)
            ),
            fare_share_rate=float(values.get("fare_share_rate", 0.0)),
            source_farebox_share_rate=float(
                values.get("source_farebox_share_rate", 0.0)
            ),
            tips_retained_fully=bool(values.get("tips_retained_fully", False)),
        )
    if "v4_legacy" not in models:
        models["v4_legacy"] = CostModel("v4_legacy", 1.0, 0.35, 0.35)
    for model in models.values():
        model.validate()
    return models


def primary_cost_model(config: Dict[str, Any]) -> CostModel:
    models = all_cost_models(config)
    name = str(config.get("cost_models", {}).get("primary", "owner_operator"))
    if name not in models:
        raise KeyError(
            f"Unknown primary cost model {name!r}; choices={sorted(models)}"
        )
    return models[name]


def raw_receipts_from_effective(
    effective_revenue: float, active_share: float
) -> float:
    active_share = float(active_share)
    if not math.isfinite(active_share) or active_share <= 0.0:
        raise ValueError(f"active_share must be positive, got {active_share}")
    return float(effective_revenue) / active_share


def operating_net_earnings(
    raw_receipts: float,
    occupied_miles: float,
    empty_miles: float,
    horizon_minutes: float,
    model: CostModel,
) -> Dict[str, float]:
    """Apply one cost model once and expose auditable accounting components."""
    model.validate()
    raw_receipts = float(raw_receipts)
    occupied_miles = float(occupied_miles)
    empty_miles = float(empty_miles)
    horizon_minutes = float(horizon_minutes)
    if min(raw_receipts, occupied_miles, empty_miles, horizon_minutes) < 0.0:
        raise ValueError("Receipts, miles, and horizon must be non-negative")

    effective_receipts = raw_receipts * model.revenue_share
    occupied_cost = occupied_miles * model.occupied_cost_per_mile
    empty_cost = empty_miles * model.empty_cost_per_mile
    mileage_cost = occupied_cost + empty_cost
    fixed_cost = horizon_minutes / 60.0 * model.fixed_lease_per_hour

    # Disclosure only. The same deduction is already embodied by revenue_share.
    disclosed_share_cost = raw_receipts * model.fare_share_rate
    net = effective_receipts - mileage_cost - fixed_cost
    identity_gap = raw_receipts - (
        effective_receipts + disclosed_share_cost
    )
    return {
        "raw_driver_receipts": raw_receipts,
        "effective_driver_receipts": effective_receipts,
        "occupied_vehicle_cost": occupied_cost,
        "empty_vehicle_cost": empty_cost,
        "mileage_cost": mileage_cost,
        "fixed_lease_cost": fixed_cost,
        "disclosed_fare_share_cost": disclosed_share_cost,
        "fare_share_identity_gap": identity_gap,
        "operating_net_earnings": net,
        # Backward-compatible alias for existing V5.1 readers. Figure text and new
        # canonical columns use "operating net earnings".
        "take_home": net,
    }


def take_home(
    raw_receipts: float,
    occupied_miles: float,
    empty_miles: float,
    horizon_minutes: float,
    model: CostModel,
) -> Dict[str, float]:
    """Backward-compatible wrapper; prefer :func:`operating_net_earnings`."""
    return operating_net_earnings(
        raw_receipts,
        occupied_miles,
        empty_miles,
        horizon_minutes,
        model,
    )
