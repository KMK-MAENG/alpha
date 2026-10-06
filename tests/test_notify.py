import urllib.parse

import pytest

from alpha.live import notify

RECORD = {
    "time": "2026-10-01T00:05:00+00:00",
    "mode": "live",
    "last_bar": "2026-09-30",
    "balances": {"equity": 2034.12, "wallet": 2010.50, "unrealized": 23.62, "available": 1500.0},
    "changes": [
        {"symbol": "ETHUSDT", "kind": "진입", "side": "롱", "quantity": 0.008, "notional": 21.4, "pnl": None},
        {"symbol": "ZILUSDT", "kind": "진입", "side": "숏", "quantity": 1400.0, "notional": 5.0, "pnl": None},
        {"symbol": "XRPUSDT", "kind": "청산", "side": "롱", "quantity": 8.2, "notional": 12.3, "pnl": 1.23, "pnl_pct": 0.10},
        {"symbol": "BNBUSDT", "kind": "증액", "side": "롱", "quantity": 0.01, "notional": 7.6, "pnl": None},
    ],
    "holdings": {"long": 40, "short": 4, "long_exposure": 0.491, "short_exposure": 0.032},
    "skipped": ["IOSTUSDT"],
    "failures": 0,
    "results": [],
    "ignored_positions": {},
}


def test_report_contains_balance_pnl_entries_and_exits():
    previous = {"balances": {"equity": 2015.72}, "time": "2026-09-30T00:05:00+00:00"}
    first = {"balances": {"equity": 2000.0}, "time": "2026-09-29T00:05:00+00:00"}
    text = notify.format_report(RECORD, previous, first)
    assert "실주문" in text and "2026-10-01 09:05" in text  # 한국 시간
    assert "2,034.12" in text and "+18.40" in text and "+34.12" in text
    assert "진입 2" in text and "ETHUSDT" in text and "숏 ZILUSDT" in text
    assert "청산 1" in text and "XRPUSDT" in text and "+1.23" in text and "+10.0%" in text
    assert "증액 1" in text
    assert "롱 40 / 숏 4" in text and "IOSTUSDT" in text


def test_report_shows_strategy_exposure_when_recorded():
    assert "🎯" not in notify.format_report(RECORD, None, None)  # 이전 기록 형식
    exposure = {"momentum": {"long": 0.21, "short": 0.05}, "carry": {"long": 0.25, "short": 0.2}}
    text = notify.format_report(RECORD | {"strategy_exposure": exposure}, None, None)
    assert "모멘텀 롱 21.0% 숏 5.0%" in text and "캐리 롱 25.0% 숏 20.0%" in text


def test_report_marks_failures():
    record = RECORD | {"failures": 1, "results": [{"order": {"symbol": "ETHUSDT"}, "error": "APIError(-2019)"}]}
    text = notify.format_report(record, None, None)
    assert "주문 실패 1" in text and "APIError(-2019)" in text


def test_send_posts_to_telegram_and_splits_long_messages(monkeypatch):
    sent = []
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda url, data, timeout: sent.append((url, data)))
    assert notify.send("가" * 5000, token="TOKEN", chat_id="42")
    assert len(sent) == 2 and sent[0][0] == "https://api.telegram.org/botTOKEN/sendMessage"
    assert urllib.parse.parse_qs(sent[0][1].decode())["chat_id"] == ["42"]


def test_send_without_settings_or_on_network_error_does_not_raise(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.send("hi") is False

    def boom(url, data, timeout):
        raise OSError("network down")

    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.send("hi", token="T", chat_id="1") is False


@pytest.mark.parametrize("mode, label", [("live", "실주문"), ("dry-run", "DRY-RUN")])
def test_failure_alert(mode, label):
    text = notify.format_failure(mode, RuntimeError("데이터가 최신이 아님"))
    assert "실행 실패" in text and label in text and "데이터가 최신이 아님" in text


def test_report_excludes_deposits_and_withdrawals_from_pnl():
    previous = {"balances": {"equity": 1034.12}, "time": "2026-09-30T00:05:00+00:00"}
    first = {"balances": {"equity": 1000.0}, "time": "2026-09-29T00:05:00+00:00"}
    text = notify.format_report(RECORD, previous, first, transfers={"previous": 1000.0, "first": 1000.0})
    # 잔고 2,034.12 중 1,000은 입금 → 전일 손익 0, 누적 +34.12 (투입 2,000 대비 +1.71%)
    assert "전일 +0.00 (+0.00%" in text and "누적 +34.12 (+1.71%" in text
    assert "입출금 +1,000.00 제외" in text
