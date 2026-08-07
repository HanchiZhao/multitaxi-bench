import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cost_models import CostModel, operating_net_earnings, take_home


def test_owner_operator_cost() -> None:
    model = CostModel("owner", 1.0, 0.70, 0.70)
    result = operating_net_earnings(100, 10, 5, 120, model)
    assert result["operating_net_earnings"] == pytest.approx(89.5)
    assert result["take_home"] == pytest.approx(89.5)


def test_fixed_lease_cost_is_subtracted_once() -> None:
    model = CostModel("fixed", 1.0, 0.25, 0.25, 9.0)
    result = take_home(100, 10, 5, 120, model)
    assert result["operating_net_earnings"] == pytest.approx(100 - 3.75 - 18)


def test_fare_share_is_not_double_counted() -> None:
    model = CostModel(
        "fare_share",
        0.6979,
        0.25,
        0.25,
        fare_share_rate=0.3021,
        source_farebox_share_rate=0.35,
        tips_retained_fully=True,
    )
    result = operating_net_earnings(100, 10, 5, 120, model)
    expected = 100 * 0.6979 - 15 * 0.25
    assert result["operating_net_earnings"] == pytest.approx(expected)
    assert result["disclosed_fare_share_cost"] == pytest.approx(30.21)
    assert result["fare_share_identity_gap"] == pytest.approx(0.0)
    # Subtracting disclosed_fare_share_cost again would be the rejected double count.
    assert result["operating_net_earnings"] != pytest.approx(expected - 30.21)


def test_inconsistent_fare_share_fields_fail() -> None:
    model = CostModel(
        "bad_fare_share", 0.6979, 0.25, 0.25, fare_share_rate=0.35
    )
    with pytest.raises(ValueError, match="revenue_share.*fare_share_rate"):
        model.validate()


def test_fixed_and_fare_share_lease_cannot_be_combined() -> None:
    model = CostModel(
        "double_lease",
        0.70,
        0.25,
        0.25,
        fixed_lease_per_hour=9.0,
        fare_share_rate=0.30,
    )
    with pytest.raises(ValueError, match="cannot be charged"):
        model.validate()
