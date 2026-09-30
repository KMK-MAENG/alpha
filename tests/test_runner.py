import json

import numpy as np
import pandas as pd
import pytest

from alpha.execution.order_manager import SymbolRules
from alpha.execution.risk import RiskError
from alpha.live import runner
from alpha.strategies import momentum
from alpha.strategies.momentum import UNIVERSE

NOW = pd.Timestamp("2026-09-29 00:05", tz="UTC")
COINS = UNIVERSE[:25]


class FakeExchange:
    def __init__(self, positions=None, last_bar="2026-09-28", one_way=True, unrealized=None):
        rng = np.random.default_rng(0)
        index = pd.date_range(end=pd.Timestamp(last_bar, tz="UTC"), periods=400, freq="1D")
        drift = np.linspace(-0.004, 0.004, len(COINS))
        rets = drift + rng.normal(0, 0.02, (len(index), len(COINS)))
        self.closes = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=index, columns=COINS)
        self._positions = pd.Series(positions or {}, dtype=float)
        self._unrealized = pd.Series(unrealized or {}, dtype=float)
        self._one_way = one_way
        self.placed = []

    def is_one_way(self):
        return self._one_way

    def rules(self):
        return {c: SymbolRules(min_notional=5.0, min_qty=0.001, step=0.001) for c in COINS}

    def daily_closes(self, symbols, days, now):
        return self.closes[symbols].iloc[-days:]

    def mark_prices(self):
        return self.closes.iloc[-1]

    def balances(self):
        return {"equity": 2000.0, "wallet": 1990.0, "unrealized": 10.0, "available": 1500.0}

    def net_transfers(self, start, end):
        return self.transfers

    transfers = 0.0

    def position_details(self):
        qty = self._positions
        return pd.DataFrame(
            {"qty": qty, "entry_price": self.closes.iloc[-1].reindex(qty.index).fillna(1.0),
             "unrealized_pnl": self._unrealized.reindex(qty.index).fillna(0.0)},
            index=qty.index,
        )

    def market_order(self, order):
        self.placed.append(order)
        return {"status": "FILLED"}


def test_dry_run_plans_orders_without_placing_and_writes_log(tmp_path):
    exchange = FakeExchange()
    record = runner.run(exchange, live=False, now=NOW, log_dir=tmp_path)
    assert record["mode"] == "dry-run" and len(record["orders"]) > 0
    assert exchange.placed == []
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["orders"] == record["orders"]


def test_live_places_all_orders_reducing_first(tmp_path):
    targets = runner.run(FakeExchange(), live=False, now=NOW, log_dir=tmp_path)["targets"]
    held = next(c for c in COINS if c not in targets)  # 목표가 0인 코인 — 보유 중이면 정리돼야 한다
    exchange = FakeExchange(positions={held: 5.0})
    record = runner.run(exchange, live=True, now=NOW, log_dir=tmp_path)
    assert [o.symbol for o in exchange.placed] == [o["symbol"] for o in record["orders"]]
    assert exchange.placed[0].symbol == held and exchange.placed[0].reduce_only
    assert record["failures"] == 0


def test_rerun_at_target_places_nothing(tmp_path):
    first = runner.run(FakeExchange(), live=False, now=NOW, log_dir=tmp_path)
    at_target = FakeExchange(positions={k: v for k, v in first["targets"].items() if v != 0})
    second = runner.run(at_target, live=True, now=NOW, log_dir=tmp_path)
    assert second["orders"] == [] and at_target.placed == []


def test_positions_outside_universe_are_left_alone(tmp_path):
    exchange = FakeExchange(positions={"PEPEUSDT": 1000.0})
    record = runner.run(exchange, live=True, now=NOW, log_dir=tmp_path)
    assert "PEPEUSDT" not in [o.symbol for o in exchange.placed]
    assert record["ignored_positions"] == {"PEPEUSDT": 1000.0}


@pytest.mark.parametrize("exchange", [FakeExchange(last_bar="2026-09-27"), FakeExchange(one_way=False)])
def test_stale_data_or_hedge_mode_aborts_without_orders(exchange, tmp_path):
    with pytest.raises(RiskError):
        runner.run(exchange, live=True, now=NOW, log_dir=tmp_path)
    assert exchange.placed == []


def test_equity_override_is_only_for_dry_run(tmp_path):
    record = runner.run(FakeExchange(), live=False, equity=1000.0, now=NOW, log_dir=tmp_path)
    assert record["equity"] == 1000.0 and record["positions"] == {}
    with pytest.raises(ValueError):
        runner.run(FakeExchange(), live=True, equity=1000.0, now=NOW, log_dir=tmp_path)


def test_runner_uses_live_strategy_options(tmp_path):
    exchange = FakeExchange()
    record = runner.run(exchange, live=False, now=NOW, log_dir=tmp_path)
    expected = momentum.combined_weights(exchange.closes, **momentum.LIVE_OPTIONS).iloc[-1]
    assert record["weights"] == pytest.approx({k: v for k, v in expected.items() if v != 0})


def test_live_opens_shorts_for_confirmed_downtrends(tmp_path):
    exchange = FakeExchange()
    record = runner.run(exchange, live=True, now=NOW, log_dir=tmp_path)
    shorts = {k for k, v in record["targets"].items() if v < 0}
    assert shorts and {o.symbol for o in exchange.placed if o.side == "SELL"} >= shorts


def test_changes_classify_entries_and_exits_with_exit_pnl(tmp_path):
    targets = runner.run(FakeExchange(), live=False, now=NOW, log_dir=tmp_path)["targets"]
    held = next(c for c in COINS if c not in targets)
    record = runner.run(FakeExchange(positions={held: 5.0}, unrealized={held: 12.5}), live=True, now=NOW, log_dir=tmp_path)
    exits = [c for c in record["changes"] if c["kind"] == "청산"]
    assert exits == [c for c in exits if c["symbol"] == held] and exits[0]["pnl"] == pytest.approx(12.5)
    entries = {c["symbol"] for c in record["changes"] if c["kind"] == "진입"}
    assert entries == set(targets)
    assert record["holdings"]["long"] + record["holdings"]["short"] == len(targets)


def test_run_with_report_sends_report(tmp_path):
    sent = []
    runner.run_with_report(FakeExchange(), live=False, equity=None, sender=sent.append, log_dir=tmp_path, now=NOW)
    assert len(sent) == 1 and "진입" in sent[0] and "DRY-RUN" in sent[0]


def test_run_with_report_alerts_on_failure_and_reraises(tmp_path):
    sent = []
    with pytest.raises(RiskError):
        runner.run_with_report(FakeExchange(last_bar="2026-09-27"), live=True, equity=None, sender=sent.append,
                               log_dir=tmp_path, now=NOW)
    assert len(sent) == 1 and "실행 실패" in sent[0]


def test_report_pnl_excludes_transfers_since_previous_run(tmp_path):
    (tmp_path / "20260928T000500_dry-run.json").write_text(
        json.dumps({"time": "2026-09-28T00:05:00+00:00", "balances": {"equity": 1000.0}})
    )
    exchange = FakeExchange()
    exchange.transfers = 1000.0  # 잔고 2,000 중 1,000은 입금
    sent = []
    runner.run_with_report(exchange, live=False, equity=None, sender=sent.append, log_dir=tmp_path, now=NOW)
    assert "전일 +0.00" in sent[0] and "입출금 +1,000.00 제외" in sent[0]


def test_cumulative_pnl_sums_stored_transfers_across_runs(tmp_path):
    for name, record in {
        "20260926T000500_dry-run.json": {"time": "2026-09-26T00:05:00+00:00", "balances": {"equity": 1500.0}},
        "20260927T000500_dry-run.json": {"time": "2026-09-27T00:05:00+00:00", "balances": {"equity": 2010.0},
                                         "transfers_since_previous": 500.0},  # 둘째 날 500 입금
    }.items():
        (tmp_path / name).write_text(json.dumps(record))
    sent = []
    runner.run_with_report(FakeExchange(), live=False, equity=None, sender=sent.append, log_dir=tmp_path, now=NOW)
    # 오늘 입출금 없음: 전일 2,010 → 2,000 = -10, 누적 1,500 → 2,000 중 500은 입금 → 0
    assert "전일 -10.00" in sent[0] and "누적 +0.00" in sent[0] and "입출금 +500.00 제외" in sent[0]


def test_failed_run_leaves_failure_record(tmp_path):
    with pytest.raises(RiskError):
        runner.run_with_report(FakeExchange(last_bar="2026-09-27"), live=True, equity=None, sender=lambda t: None,
                               log_dir=tmp_path, now=NOW)
    failed = json.loads(next(tmp_path.glob("*_live_failed.json")).read_text())
    assert failed["mode"] == "live" and "RiskError" in failed["error"]
    assert runner._records(tmp_path, "live", "2099") == []  # 실패 기록은 손익 기준에 쓰지 않는다
