"""주문 전 안전 점검. 하나라도 걸리면 RiskError로 주문 없이 중단한다.

상한값들은 전략상 나올 수 없는 수준이다 (결합 50:50의 과거 최대 총 노출은 약 1.24배) — 버그·데이터 이상을 막는 용도.
"""
import pandas as pd

from alpha.execution.order_manager import Order

MAX_GROSS = 1.5  # 총 목표 노출 (|비중| 합)
MAX_ORDER_FRACTION = 0.5  # 주문 1건 / 평가금액


class RiskError(RuntimeError):
    pass


def check_data_fresh(last_bar: pd.Timestamp, now: pd.Timestamp) -> None:
    expected = now.floor("D") - pd.Timedelta(days=1)
    if last_bar != expected:
        raise RiskError(f"데이터가 최신이 아님: 마지막 일봉 {last_bar.date()}, 기대 {expected.date()}")


def check_weights(weights: pd.Series) -> None:
    gross = weights.abs().sum()
    if gross > MAX_GROSS:
        raise RiskError(f"총 목표 노출 {gross:.2f}배가 상한 {MAX_GROSS}배 초과")


def check_orders(orders: list[Order], equity: float) -> None:
    if equity <= 0:
        raise RiskError(f"평가금액이 0 이하: {equity}")
    for order in orders:
        if order.notional > MAX_ORDER_FRACTION * equity:
            raise RiskError(f"주문 1건이 평가금액의 {MAX_ORDER_FRACTION:.0%} 초과: {order}")
