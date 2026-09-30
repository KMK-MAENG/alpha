import json

import pandas as pd
import pytest

from alpha.live import bot

NOW = pd.Timestamp("2026-10-01 03:00", tz="UTC")  # 한국 12:00


class FakeExchange:
    transfers = 0.0

    def balances(self):
        return {"equity": 2034.12, "wallet": 2010.50, "unrealized": 23.62, "available": 1500.0}

    def position_details(self):
        return pd.DataFrame(
            {"qty": [0.5, -100.0, 3.0], "entry_price": [2000.0, 1.0, 10.0], "unrealized_pnl": [50.0, -5.0, 1.0]},
            index=["ETHUSDT", "ZILUSDT", "PEPEUSDT"],
        )

    def mark_prices(self):
        return pd.Series({"ETHUSDT": 2100.0, "ZILUSDT": 1.05, "PEPEUSDT": 10.33})

    def net_transfers(self, start, end):
        return self.transfers


def write(log_dir, name, record):
    (log_dir / name).write_text(json.dumps(record))


def test_balance():
    text = bot.handle("/balance", FakeExchange(), log_dir=None, now=NOW)
    assert "2,034.12" in text and "+23.62" in text and "1,500.00" in text


def test_positions_show_side_pnl_weight_and_outside_strategy():
    text = bot.handle("/positions", FakeExchange(), log_dir=None, now=NOW)
    assert "롱 ETHUSDT" in text and "+50.00" in text and "+5.0%" in text  # 50 / (0.5 × 2000)
    assert "숏 ZILUSDT" in text and "-5.00" in text
    assert "51.6%" in text  # 0.5 × 2100 / 2034.12
    assert "PEPEUSDT" in text and "전략 밖" in text
    assert "롱 2 / 숏 1" in text


def test_pnl_uses_live_records_and_excludes_transfers(tmp_path):
    write(tmp_path, "20260929T000500_live.json", {"time": "2026-09-29T00:05:00+00:00", "balances": {"equity": 1000.0}})
    write(tmp_path, "20260930T000500_live.json", {"time": "2026-09-30T00:05:00+00:00", "balances": {"equity": 1010.0},
                                                   "transfers_since_previous": 0.0})
    exchange = FakeExchange()
    exchange.transfers = 1000.0  # 마지막 실행 이후 1,000 입금
    text = bot.handle("/pnl", exchange, log_dir=tmp_path, now=NOW)
    assert "+24.12" in text and "+34.12" in text and "입출금 +1,000.00 제외" in text


def test_pnl_without_live_records(tmp_path):
    assert "실주문 기록이 아직 없음" in bot.handle("/pnl", FakeExchange(), log_dir=tmp_path, now=NOW)


def test_last_rerenders_latest_report(tmp_path):
    record = {
        "time": "2026-10-01T00:05:00+00:00", "mode": "live", "last_bar": "2026-09-30",
        "balances": {"equity": 2034.12}, "transfers_since_previous": 0.0,
        "changes": [{"symbol": "ETHUSDT", "kind": "진입", "side": "롱", "quantity": 0.008, "notional": 21.4, "pnl": None}],
        "holdings": {"long": 1, "short": 0, "long_exposure": 0.01, "short_exposure": 0.0},
        "skipped": [], "failures": 0, "results": [], "ignored_positions": {},
    }
    write(tmp_path, "20261001T000500_live.json", record)
    text = bot.handle("/last", FakeExchange(), log_dir=tmp_path, now=NOW)
    assert "진입 1" in text and "ETHUSDT" in text


def test_status_shows_last_runs_failure_and_next_run(tmp_path):
    write(tmp_path, "20260930T000500_live.json", {"time": "2026-09-30T00:05:00+00:00", "mode": "live",
                                                   "balances": {"equity": 1.0}, "orders": [{}] * 3, "failures": 0})
    write(tmp_path, "20261001T000500_live_failed.json", {"time": "2026-10-01T00:05:00+00:00", "mode": "live",
                                                          "error": "RiskError: 데이터가 최신이 아님"})
    text = bot.handle("/status", FakeExchange(), log_dir=tmp_path, now=NOW)
    assert "09-30 09:05" in text and "주문 3" in text
    assert "실패" in text and "데이터가 최신이 아님" in text
    assert "10-02 09:05" in text  # 다음 실행 (한국 시간)


def test_unknown_command_and_handler_errors():
    assert "/balance" in bot.handle("/hello", FakeExchange(), log_dir=None, now=NOW)

    class Broken(FakeExchange):
        def balances(self):
            raise ConnectionError("binance down")

    assert "오류" in bot.handle("/balance", Broken(), log_dir=None, now=NOW)


def test_process_updates_answers_only_configured_chat():
    replies = []
    updates = [
        {"update_id": 10, "message": {"chat": {"id": 42}, "text": "/balance"}},
        {"update_id": 11, "message": {"chat": {"id": 999}, "text": "/balance"}},  # 모르는 사람
        {"update_id": 12, "edited_message": {"chat": {"id": 42}, "text": "x"}},
    ]
    offset = bot.process_updates(updates, chat_id="42", reply=replies.append, answer=lambda text: f"답:{text}")
    assert offset == 13 and replies == ["답:/balance"]
