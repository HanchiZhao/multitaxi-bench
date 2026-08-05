"""Cost-model configuration and post-hoc accounting for V5.1.

The V4 simulator remains the operational core. This module supplies explicit cost models
without replacing V4's dynamics, EB estimation, policies, scenario bank, or validations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import yaml

@dataclass(frozen=True)
class CostModel:
    name: str
    revenue_share: float
    occupied_cost_per_mile: float
    empty_cost_per_mile: float
    fixed_lease_per_hour: float = 0.0
    fare_share_rate: float = 0.0

    def validate(self) -> None:
        if not 0 < self.revenue_share <= 1:
            raise ValueError(f"revenue_share must be in (0,1], got {self.revenue_share}")
        for value, label in [
            (self.occupied_cost_per_mile, "occupied_cost_per_mile"),
            (self.empty_cost_per_mile, "empty_cost_per_mile"),
            (self.fixed_lease_per_hour, "fixed_lease_per_hour"),
            (self.fare_share_rate, "fare_share_rate"),
        ]:
            if value < 0:
                raise ValueError(f"{label} must be non-negative")

    def env(self) -> Dict[str, str]:
        self.validate()
        return {
            "MULTITAXI_COST_MODEL": self.name,
            "MULTITAXI_DRIVER_REVENUE_SHARE": str(self.revenue_share),
            "MULTITAXI_OCCUPIED_COST_PER_MILE": str(self.occupied_cost_per_mile),
            "MULTITAXI_EMPTY_COST_PER_MILE": str(self.empty_cost_per_mile),
            "MULTITAXI_FIXED_LEASE_PER_HOUR": str(self.fixed_lease_per_hour),
            "MULTITAXI_FARE_SHARE_RATE": str(self.fare_share_rate),
        }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p=Path(path)
    data=yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data,dict): raise ValueError("Config must be a YAML mapping")
    return data


def all_cost_models(config: Dict[str, Any]) -> Dict[str, CostModel]:
    raw=config.get("cost_models",{})
    models={}
    for name, values in raw.items():
        if name=="primary" or not isinstance(values,dict): continue
        models[name]=CostModel(
            name=name,
            revenue_share=float(values.get("revenue_share",1.0)),
            occupied_cost_per_mile=float(values.get("occupied_cost_per_mile",0.35)),
            empty_cost_per_mile=float(values.get("empty_cost_per_mile",0.35)),
            fixed_lease_per_hour=float(values.get("fixed_lease_per_hour",0.0)),
            fare_share_rate=float(values.get("fare_share_rate",0.0)),
        )
    if "v4_legacy" not in models:
        models["v4_legacy"]=CostModel("v4_legacy",1.0,0.35,0.35)
    for m in models.values(): m.validate()
    return models


def primary_cost_model(config: Dict[str, Any]) -> CostModel:
    models=all_cost_models(config)
    name=str(config.get("cost_models",{}).get("primary","owner_operator"))
    if name not in models: raise KeyError(f"Unknown primary cost model {name!r}; choices={sorted(models)}")
    return models[name]


def raw_receipts_from_effective(effective_revenue: float, active_share: float) -> float:
    return float(effective_revenue)/max(float(active_share),1e-12)


def take_home(raw_receipts: float, occupied_miles: float, empty_miles: float,
              horizon_minutes: float, model: CostModel) -> Dict[str,float]:
    effective_receipts=float(raw_receipts)*model.revenue_share
    mileage_cost=float(occupied_miles)*model.occupied_cost_per_mile + float(empty_miles)*model.empty_cost_per_mile
    fixed_cost=float(horizon_minutes)/60.0*model.fixed_lease_per_hour
    # fare_share_rate is reported explicitly. If revenue_share already equals 1-share,
    # do not subtract it again from take-home.
    disclosed_share_cost=float(raw_receipts)*model.fare_share_rate
    net=effective_receipts-mileage_cost-fixed_cost
    return {
        "raw_driver_receipts":float(raw_receipts),
        "effective_driver_receipts":effective_receipts,
        "occupied_vehicle_cost":float(occupied_miles)*model.occupied_cost_per_mile,
        "empty_vehicle_cost":float(empty_miles)*model.empty_cost_per_mile,
        "mileage_cost":mileage_cost,
        "fixed_lease_cost":fixed_cost,
        "disclosed_fare_share_cost":disclosed_share_cost,
        "take_home":net,
    }
