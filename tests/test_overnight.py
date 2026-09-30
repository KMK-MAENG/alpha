import numpy as np
import pandas as pd
import pytest

from alpha.backtest import overnight

UTC = lambda *xs: pd.to_datetime(list(xs), utc=True)

# 월(6/1)·화(6/2) 정규장 다음은 다음 주 월(6/8) — 화요일 마감 뒤는 주말·휴장 구간
SCHEDULE = pd.DataFrame(
    {
        "open": UTC("2026-06-01 13:30", "2026-06-02 13:30", "2026-06-08 13:30"),
        "close": UTC("2026-06-01 20:00", "2026-06-02 20:00", "2026-06-08 20:00"),
    }
)


def test_label_sessions_at_open_close_boundaries():
    index = pd.DatetimeIndex(
        UTC("2026-06-01 13:30", "2026-06-01 19:59", "2026-06-01 20:00", "2026-06-02 13:29", "2026-06-02 20:00")
    )
    assert overnight.label_sessions(index, SCHEDULE).tolist() == [
        "rth", "rth", "weeknight", "weeknight", "weekend_holiday"
    ]


def test_backtest_splits_price_funding_and_fee():
    index = pd.date_range("2026-06-01 19:58", periods=4, freq="1min", tz="UTC")  # 19:58 ~ 20:01
    candles = pd.DataFrame({"close": [100.0, 100.0, 101.0, 102.01]}, index=index)
    pos = pd.Series([0.0, 0.0, 1.0, 1.0], index=index)  # 20:00 봉부터 롱
    funding = pd.Series([0.001], index=UTC("2026-06-01 20:01:00.001"))

    pnl = overnight.backtest(candles, funding, pos, fee=0.0005)

    assert pnl["price"].tolist() == pytest.approx([0.0, 0.0, 0.01, 0.01])
    assert pnl["funding"].tolist() == pytest.approx([0.0, 0.0, 0.0, -0.001])
    assert pnl["fee"].tolist() == pytest.approx([0.0, 0.0, -0.0005, 0.0])
    assert np.allclose(pnl["total"], pnl["price"] + pnl["funding"] + pnl["fee"])


def test_session_dates_map_each_overnight_bar_to_preceding_close():
    index = pd.DatetimeIndex(UTC("2026-06-01 13:30", "2026-06-01 20:00", "2026-06-02 13:29", "2026-06-02 21:00"))
    schedule = SCHEDULE.set_index(pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-08"]))
    assert overnight.session_dates(index, schedule).strftime("%m-%d").tolist() == ["06-01", "06-01", "06-01", "06-02"]
