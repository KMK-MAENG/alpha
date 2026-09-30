"""목표 비중을 거래소 규칙에 맞는 주문 목록으로 바꾼다 (순수 함수, 네트워크 없음).

- 목표 금액 = 비중 × 계좌 평가금액. 거래소 최소 주문(최소 명목 금액, 최소 수량 × 가격 중 큰 값) 대비
  절반 미만이면 0, 절반 이상 최소 미만이면 최소 크기로 올리고, 그 이상이면 수량 단위로 0 방향 버림
- 주문 = 목표 수량 - 현재 수량. 최소 주문 금액 미만 차이는 건너뛰되, 0으로 정리하는 주문은 reduce-only로 낸다
- 거래 불가 코인(rules에 없음)의 목표는 0. 줄이는 주문을 먼저 내 증거금을 확보한다
"""
import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SymbolRules:
    min_notional: float
    min_qty: float
    step: float


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float
    reduce_only: bool
    notional: float  # 주문 시점 가격 기준 참고 금액


def _decimals(step: float) -> int:
    return max(0, -math.floor(math.log10(step) + 1e-9))


def _floor_to_step(qty: float, step: float) -> float:
    return round(math.floor(qty / step + 1e-9) * step, _decimals(step))


def _ceil_to_step(qty: float, step: float) -> float:
    return round(math.ceil(qty / step - 1e-9) * step, _decimals(step))


def target_quantities(
    weights: pd.Series, equity: float, prices: pd.Series, rules: dict[str, SymbolRules]
) -> pd.Series:
    targets = {}
    for symbol, weight in weights.fillna(0.0).items():
        rule, price = rules.get(symbol), prices.get(symbol)
        notional = weight * equity if equity > 0 else 0.0
        if rule is None or not price or notional == 0:
            targets[symbol] = 0.0
            continue
        need = max(rule.min_notional, rule.min_qty * price)
        if abs(notional) < need / 2:
            targets[symbol] = 0.0
            continue
        qty = _floor_to_step(abs(notional) / price, rule.step)
        if qty * price < need:  # 최소 크기로 올린다 (반올림 규칙 또는 버림으로 최소 미만이 된 경우)
            qty = _ceil_to_step(need / price, rule.step)
        targets[symbol] = math.copysign(qty, notional)
    return pd.Series(targets, dtype=float)


def plan_orders(
    targets: pd.Series, positions: pd.Series, prices: pd.Series, rules: dict[str, SymbolRules]
) -> tuple[list[Order], list[str]]:
    """(주문 목록 — 줄이는 주문 먼저, 최소 주문 금액 미만이라 건너뛴 코인)."""
    orders, skipped = [], []
    for symbol in targets.index.union(positions.index):
        target, current = float(targets.get(symbol, 0.0)), float(positions.get(symbol, 0.0))
        rule = rules.get(symbol)
        step = rule.step if rule else 10.0 ** -_decimals_of(current)
        delta = round(target - current, _decimals(step))
        if delta == 0:
            continue
        price = float(prices.get(symbol, float("nan")))
        closing = bool(target == 0)
        notional = abs(delta) * price
        if not closing and (rule is None or notional < rule.min_notional):
            skipped.append(symbol)
            continue
        orders.append(
            Order(symbol, "BUY" if delta > 0 else "SELL", abs(delta), reduce_only=closing, notional=round(notional, 8))
        )
    reducing = lambda o: abs(targets.get(o.symbol, 0.0)) < abs(positions.get(o.symbol, 0.0))
    return sorted(orders, key=lambda o: not reducing(o)), skipped


def _decimals_of(qty: float) -> int:
    """거래 규칙이 없는(거래 불가) 코인의 보유 수량 자릿수 — 전량 정리 주문용."""
    text = f"{abs(qty):.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0
