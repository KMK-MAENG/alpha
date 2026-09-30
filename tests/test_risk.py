import pandas as pd
import pytest

from alpha.execution import risk
from alpha.execution.order_manager import Order

NOW = pd.Timestamp("2026-09-29 00:05", tz="UTC")


def test_data_is_fresh_only_when_last_bar_is_yesterday():
    risk.check_data_fresh(pd.Timestamp("2026-09-28", tz="UTC"), NOW)
    with pytest.raises(risk.RiskError, match="최신"):
        risk.check_data_fresh(pd.Timestamp("2026-09-27", tz="UTC"), NOW)


def test_gross_exposure_cap():
    risk.check_weights(pd.Series({"A": 0.8, "B": -0.6}))
    with pytest.raises(risk.RiskError, match="노출"):
        risk.check_weights(pd.Series({"A": 1.0, "B": -0.6}))


def test_order_size_cap_and_positive_equity():
    small = Order("A", "BUY", 1.0, reduce_only=False, notional=400.0)
    risk.check_orders([small], equity=1000.0)
    with pytest.raises(risk.RiskError, match="주문"):
        risk.check_orders([Order("A", "BUY", 1.0, reduce_only=False, notional=600.0)], equity=1000.0)
    with pytest.raises(risk.RiskError, match="평가금액"):
        risk.check_orders([], equity=0.0)
