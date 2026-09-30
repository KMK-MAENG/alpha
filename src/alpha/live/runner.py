"""하루 1회 실행: 결합 50:50(진입 확인 + 시계열 숏, momentum.LIVE_OPTIONS) 목표 비중을 계산해 바이낸스 USDT-M 선물 포지션을 맞춘다.

매일 UTC 00:05(한국 09:05)에 실행한다. 기본은 dry-run(계산·기록만), --live를 명시해야 실주문.
전략 코인(UNIVERSE) 밖의 포지션은 건드리지 않는다. 다시 실행해도 목표와의 차이만 주문한다.
기록: logs/live/<UTC 시각>_<모드>.json. 실행 결과는 텔레그램으로 보고하고, 실패하면 경고를 보낸다 (live/notify.py).
설계: docs/superpowers/specs/2026-09-29-momentum-live-trading-design.md

실행 (프로젝트 루트에서):
  uv run python -m alpha.live.runner --setup         # 교차 증거금·레버리지 설정 (최초 1회)
  uv run python -m alpha.live.runner                 # dry-run (실계좌 조회, 주문 없음)
  uv run python -m alpha.live.runner --equity 2000   # API 키 없이 미리보기 (포지션 없음 가정)
  uv run python -m alpha.live.runner --live          # 실주문
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from alpha.config import get_env
from alpha.exchange.client import BinanceFutures
from alpha.execution import risk
from alpha.execution.order_manager import Order, plan_orders, target_quantities
from alpha.live import notify
from alpha.strategies import momentum

LOOKBACK_DAYS = 300  # 신호(120일) + 변동성(60일) + 변동성 타게팅(60일)에 충분한 길이
LEVERAGE = 3  # 거래소 레버리지 설정 (증거금 여유용, 실제 노출은 전략 비중이 정한다)
LOG_DIR = Path("logs/live")


def run(exchange, live: bool, equity: float | None = None, now: pd.Timestamp | None = None, log_dir: Path = LOG_DIR) -> dict:
    """equity를 주면 계좌 조회 없이 그 평가금액·무포지션으로 계산만 한다 (dry-run 전용 미리보기)."""
    if live and equity is not None:
        raise ValueError("--equity 미리보기는 dry-run 전용이다")
    now = now or pd.Timestamp.now(tz="UTC")
    use_account = equity is None
    if use_account and not exchange.is_one_way():
        raise risk.RiskError("헤지 모드 계좌 — 단방향(one-way) 모드로 바꿔야 한다")

    rules = exchange.rules()
    closes = exchange.daily_closes([s for s in momentum.UNIVERSE if s in rules], LOOKBACK_DAYS, now)
    risk.check_data_fresh(closes.index[-1], now)
    weights = momentum.combined_weights(closes, **momentum.LIVE_OPTIONS).iloc[-1]
    risk.check_weights(weights)

    if use_account:
        balances, details = exchange.balances(), exchange.position_details()
        equity = balances["equity"]
    else:
        balances = {"equity": equity}
        details = pd.DataFrame(columns=["qty", "entry_price", "unrealized_pnl"], dtype=float)
    all_positions = details["qty"]
    in_universe = all_positions.index.isin(momentum.UNIVERSE)
    positions, ignored = all_positions[in_universe], all_positions[~in_universe]
    prices = exchange.mark_prices()
    targets = target_quantities(weights, equity, prices, rules)
    orders, skipped = plan_orders(targets, positions, prices, rules)
    risk.check_orders(orders, equity)

    results = []
    for order in orders if live else []:
        try:
            results.append({"order": asdict(order), "response": exchange.market_order(order)})
        except Exception as e:  # 한 주문이 실패해도 나머지는 계속 내고, 실패는 기록·종료 코드로 알린다
            results.append({"order": asdict(order), "error": repr(e)})

    changes, after = _changes(orders, positions, details, results)
    earlier = _records(log_dir, "live" if live else "dry-run", now.isoformat())
    # 직전 실행 이후 입출금 — 매번 짧은 기간만 조회해 저장하고, 누적은 저장된 값을 합산한다
    transfers = exchange.net_transfers(pd.Timestamp(earlier[-1]["time"]), now) if use_account and earlier else 0.0
    record = {
        "time": now.isoformat(),
        "mode": "live" if live else "dry-run",
        "last_bar": str(closes.index[-1].date()),
        "equity": equity,
        "balances": balances,
        "transfers_since_previous": transfers,
        "changes": changes,
        "holdings": {
            "long": int((after > 0).sum()),
            "short": int((after < 0).sum()),
            "long_exposure": float((after.clip(lower=0) * prices.reindex(after.index)).sum() / equity),
            "short_exposure": float((-after.clip(upper=0) * prices.reindex(after.index)).sum() / equity),
        },
        "gross_target": float(weights.abs().sum()),
        "weights": {k: float(v) for k, v in weights.items() if v != 0},
        "targets": {k: float(v) for k, v in targets.items() if v != 0},
        "positions": {k: float(v) for k, v in positions.items()},
        "ignored_positions": {k: float(v) for k, v in ignored.items()},
        "orders": [asdict(o) for o in orders],
        "skipped": skipped,
        "results": results,
        "failures": sum("error" in r for r in results),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{now:%Y%m%dT%H%M%S}_{record['mode']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    return record


def _changes(orders: list[Order], positions: pd.Series, details: pd.DataFrame, results: list[dict]):
    """주문(실주문이면 성공한 것만)으로 바뀐 포지션을 진입·청산·전환·증액·감액으로 분류하고, 주문 후 보유 수량을 돌려준다.
    줄이거나 닫는 주문의 손익은 그 부분의 청산 직전 미실현 손익으로 본다."""
    failed = {r["order"]["symbol"] for r in results if "error" in r}
    after = positions.astype(float).copy()
    changes = []
    for o in orders:
        if o.symbol in failed:
            continue
        before = float(positions.get(o.symbol, 0.0))
        delta = o.quantity if o.side == "BUY" else -o.quantity
        now = round(before + delta, 10)
        after[o.symbol] = now
        side = lambda q: "롱" if q > 0 else "숏"
        if before == 0:
            kind, label = "진입", side(now)
        elif now == 0:
            kind, label = "청산", side(before)
        elif (before > 0) != (now > 0):
            kind, label = "전환", f"{side(before)}→{side(now)}"
        else:
            kind, label = ("증액" if abs(now) > abs(before) else "감액"), side(before)
        change = {"symbol": o.symbol, "kind": kind, "side": label, "quantity": o.quantity, "notional": o.notional, "pnl": None}
        if kind != "진입" and kind != "증액" and o.symbol in details.index:
            closed = min(abs(delta), abs(before)) / abs(before)
            row = details.loc[o.symbol]
            change["pnl"] = float(row["unrealized_pnl"] * closed)
            cost = abs(before) * closed * row["entry_price"]
            change["pnl_pct"] = change["pnl"] / cost if cost else None
        changes.append(change)
    return changes, after[after != 0]


def _records(log_dir: Path, mode: str, before: str) -> list[dict]:
    """같은 모드에서 before 시각 이전의 실행 기록 (시간순, 잔고가 있는 것만)."""
    records = [json.loads(p.read_text()) for p in sorted(log_dir.glob(f"*_{mode}.json"))]
    return [r for r in records if r["time"] < before and "balances" in r]


def run_with_report(exchange, live: bool, equity: float | None, sender=notify.send, log_dir: Path = LOG_DIR,
                    now: pd.Timestamp | None = None) -> dict:
    """run()을 실행하고 결과를 보고한다. 실패하면 경고를 보낸 뒤 예외를 그대로 올린다."""
    mode = "live" if live else "dry-run"
    try:
        record = run(exchange, live=live, equity=equity, now=now, log_dir=log_dir)
    except Exception as e:
        failed_at = now or pd.Timestamp.now(tz="UTC")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{failed_at:%Y%m%dT%H%M%S}_{mode}_failed.json").write_text(
            json.dumps({"time": failed_at.isoformat(), "mode": mode, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        )
        sender(notify.format_failure(mode, e))
        raise
    sender(build_report(record, log_dir))
    return record


def build_report(record: dict, log_dir: Path) -> str:
    """실행 기록의 보고 문구. 손익은 같은 모드의 직전·첫 실행 대비이며 저장된 입출금을 뺀다."""
    earlier = _records(log_dir, record["mode"], record["time"])
    previous, first = (earlier[-1], earlier[0]) if earlier else (None, None)
    own = record.get("transfers_since_previous", 0.0)
    transfers = {"previous": own, "first": own + sum(r.get("transfers_since_previous", 0.0) for r in earlier[1:])}
    return notify.format_report(record, previous, first, transfers)


def _print(record: dict) -> None:
    print(f"[{record['mode']}] {record['time']} | 마지막 일봉 {record['last_bar']} | 평가금액 {record['equity']:,.2f} USDT"
          f" | 목표 총 노출 {record['gross_target']:.1%} ({len(record['targets'])}개 코인)")
    for o in record["orders"]:
        print(f"  {o['side']:4} {o['symbol']:10} {o['quantity']:>14} ≈ {o['notional']:>9,.2f} USDT"
              f"{' (reduce-only)' if o['reduce_only'] else ''}")
    if record["skipped"]:
        print(f"  최소 주문 금액 미만이라 건너뜀: {', '.join(record['skipped'])}")
    if record["ignored_positions"]:
        print(f"  전략 밖 포지션(건드리지 않음): {record['ignored_positions']}")
    for r in record["results"]:
        if "error" in r:
            print(f"  주문 실패 {r['order']['symbol']}: {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="실제 주문을 낸다 (없으면 dry-run)")
    parser.add_argument("--setup", action="store_true", help="교차 증거금·레버리지 설정")
    parser.add_argument("--equity", type=float, help="API 키 없이 이 평가금액으로 미리보기 (dry-run 전용)")
    args = parser.parse_args()
    if args.live and args.equity is not None:
        parser.error("--equity 미리보기는 dry-run 전용이라 --live와 함께 쓸 수 없다")

    if args.equity is not None:
        exchange = BinanceFutures()
    else:
        key, secret = get_env("BINANCE_API_KEY"), get_env("BINANCE_API_SECRET")
        if not key or not secret:
            sys.exit(".env에 BINANCE_API_KEY, BINANCE_API_SECRET을 설정해야 한다 (키 없이 보려면 --equity)")
        exchange = BinanceFutures(key, secret)

    if args.setup:
        symbols = [s for s in momentum.UNIVERSE if s in exchange.rules()]
        exchange.setup(symbols, LEVERAGE)
        print(f"{len(symbols)}개 코인: 교차 증거금, 레버리지 {LEVERAGE}배 설정 완료")
        return

    record = run_with_report(exchange, live=args.live, equity=args.equity)
    _print(record)
    sys.exit(1 if record["failures"] else 0)


if __name__ == "__main__":
    main()
