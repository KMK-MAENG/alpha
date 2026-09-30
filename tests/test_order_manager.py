import pandas as pd
import pytest

from alpha.execution.order_manager import Order, SymbolRules, plan_orders, target_quantities

RULES = {
    "AAAUSDT": SymbolRules(min_notional=5.0, min_qty=0.001, step=0.001),
    "BBBUSDT": SymbolRules(min_notional=20.0, min_qty=0.1, step=0.1),
}
PRICES = pd.Series({"AAAUSDT": 100.0, "BBBUSDT": 10.0})


def targets(weights: dict, equity: float = 1000.0) -> pd.Series:
    return target_quantities(pd.Series(weights, dtype=float), equity, PRICES, RULES)


def test_target_quantity_rounds_toward_zero_on_step():
    # 0.0333 × 1000 / 100 = 0.333 개 → 단위 0.001 → 0.333
    assert targets({"AAAUSDT": 0.0333})["AAAUSDT"] == 0.333
    # 음수(숏)도 0 방향으로 버림
    assert targets({"BBBUSDT": -0.0337})["BBBUSDT"] == -3.3


def test_target_below_half_minimum_is_zero_and_between_is_rounded_up_to_minimum():
    # AAA 최소 5 USDT: 2 USDT(절반 미만) → 0, 3 USDT(절반 이상) → 최소 크기 0.05개
    assert targets({"AAAUSDT": 0.002})["AAAUSDT"] == 0.0
    assert targets({"AAAUSDT": 0.003})["AAAUSDT"] == 0.05
    # BBB 최소 20 USDT, 수량 단위 0.1: 15 USDT → 2.0개 (= 20 USDT)
    assert targets({"BBBUSDT": 0.015})["BBBUSDT"] == 2.0


def test_untradable_symbol_gets_zero_target():
    t = target_quantities(pd.Series({"CCCUSDT": 0.1}), 1000.0, PRICES, RULES)
    assert t.get("CCCUSDT", 0.0) == 0.0


def test_plan_orders_buys_and_sells_the_difference():
    orders, skipped = plan_orders(
        pd.Series({"AAAUSDT": 0.5, "BBBUSDT": 5.0}),
        pd.Series({"AAAUSDT": 0.2, "BBBUSDT": 8.0}),
        PRICES,
        RULES,
    )
    assert orders == [
        Order("BBBUSDT", "SELL", 3.0, reduce_only=False, notional=30.0),  # 줄이는 주문이 먼저
        Order("AAAUSDT", "BUY", 0.3, reduce_only=False, notional=30.0),
    ]
    assert skipped == []


def test_plan_orders_skips_differences_below_minimum_but_always_closes():
    orders, skipped = plan_orders(
        pd.Series({"AAAUSDT": 0.52, "BBBUSDT": 0.0}),
        pd.Series({"AAAUSDT": 0.5, "BBBUSDT": 1.0, "CCCUSDT": 7.0}),  # CCC는 거래 불가인데 보유 중
        pd.Series({**PRICES, "CCCUSDT": 1.0}),
        RULES,
    )
    assert skipped == ["AAAUSDT"]  # 0.02개 × 100 = 2 USDT < 5
    assert Order("BBBUSDT", "SELL", 1.0, reduce_only=True, notional=10.0) in orders  # 10 USDT < 20이어도 정리
    assert Order("CCCUSDT", "SELL", 7.0, reduce_only=True, notional=7.0) in orders


def test_plan_orders_quantities_are_exact_multiples_of_step():
    orders, _ = plan_orders(pd.Series({"AAAUSDT": 0.3}), pd.Series({"AAAUSDT": 0.1}), PRICES, RULES)
    assert orders[0].quantity == 0.2 and repr(orders[0].quantity) == "0.2"


def test_flip_from_long_to_short_is_single_order():
    orders, _ = plan_orders(pd.Series({"BBBUSDT": -3.0}), pd.Series({"BBBUSDT": 2.0}), PRICES, RULES)
    assert orders == [Order("BBBUSDT", "SELL", 5.0, reduce_only=False, notional=50.0)]


@pytest.mark.parametrize("equity", [0.0, -10.0])
def test_non_positive_equity_means_no_targets(equity):
    assert (targets({"AAAUSDT": 0.5}, equity) == 0).all()
