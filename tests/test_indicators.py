import numpy as np
import pandas as pd
import pytest

from alpha.strategies import indicators as ind


def daily(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC"), dtype=float)


def random_walk(n=600, seed=0):
    return daily(100 * np.cumprod(1 + np.random.default_rng(seed).normal(0, 0.03, n)))


def test_rsi_bounds_and_extremes():
    assert ind.rsi(daily(np.arange(1, 60))).iloc[-1] == pytest.approx(100.0)  # 계속 오르면 100
    assert ind.rsi(daily(np.arange(60, 1, -1))).iloc[-1] == pytest.approx(0.0)  # 계속 내리면 0
    r = ind.rsi(random_walk()).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_sma_cross_and_price_vs_sma():
    up = daily(np.linspace(100, 300, 400))
    assert ind.SIGNALS["sma_cross_50_200"](up).iloc[-1] == 1.0
    assert ind.SIGNALS["price_vs_sma200"](up).iloc[-1] == 1.0
    assert ind.SIGNALS["sma_cross_50_200"](daily(np.linspace(300, 100, 400))).iloc[-1] == -1.0


def test_macd_sign_follows_acceleration():
    accelerating = daily(100 * np.exp(np.linspace(0, 1, 200) ** 2))
    assert ind.SIGNALS["macd_12_26_9"](accelerating).iloc[-1] == 1.0


def test_donchian_enters_on_breakout_and_exits_on_channel_break():
    # 55일 최고가 돌파(110) → 유지 → 20일 최저가(111)는 깨지만 55일 최저가(100)는 안 깨는 105 → 청산
    prices = [80.0] * 5 + [100.0] * 55 + [110.0] + [111.0] * 25 + [105.0]
    s = ind.SIGNALS["donchian_55_20"](daily(prices))
    assert s.iloc[59] == 0 and s.iloc[60] == 1 and s.iloc[85] == 1 and s.iloc[86] == 0
    assert ind.SIGNALS["donchian_55_20"](daily(prices[:86] + [95.0])).iloc[86] == -1  # 55일 최저가 이탈은 숏


def test_rsi_reversal_buys_oversold_and_exits_above_50():
    r = daily([50, 25, 40, 55, 60, 75, 60, 45])
    s = ind.rsi_reversal_positions(r)
    assert s.tolist() == [0, 1, 1, 0, 0, -1, -1, 0]


@pytest.mark.parametrize("name", list(ind.SIGNALS) + list(ind.ENSEMBLES))
def test_signals_use_only_past_data(name):
    close = random_walk()
    full = ind.signal(close, name).iloc[:400]
    head = ind.signal(close.iloc[:400], name)
    pd.testing.assert_series_equal(full, head, check_names=False)
    assert full.dropna().between(-1, 1).all()


def test_bollinger_breakout_and_reversion_are_mirror_rules():
    # 잔잔하다가 상단 돌파(130) → 중심선 위 유지(125) → 중심선 아래(95)
    prices = [100.0, 101.0] * 15 + [130.0, 125.0, 95.0]
    breakout = ind.SIGNALS["bb_breakout_20_2"](daily(prices))
    reversion = ind.SIGNALS["bb_reversion_20_2"](daily(prices))
    assert breakout.iloc[29] == 0 and breakout.iloc[30] == 1 and breakout.iloc[31] == 1 and breakout.iloc[32] == 0
    assert reversion.iloc[30] == -1  # 상단 돌파는 과열 → 숏
    # 하단 이탈(70) → 롱, 중심선 위로 회복(110) → 청산
    down = [100.0, 101.0] * 15 + [70.0, 90.0, 110.0]
    r = ind.SIGNALS["bb_reversion_20_2"](daily(down))
    assert r.iloc[30] == 1 and r.iloc[31] == 1 and r.iloc[32] == 0


def test_bollinger_percent_b_is_bounded_trend_position():
    up = daily(np.linspace(100, 200, 100))
    z = ind.SIGNALS["bb_percent_b_20"](up)
    assert z.iloc[-1] > 0 and z.dropna().between(-1, 1).all()


def test_ensemble_is_equal_weight_mean_of_members():
    close = random_walk()
    members = ind.ENSEMBLES["ens_diverse_3"]
    expected = pd.concat([ind.SIGNALS[m](close) for m in members], axis=1).mean(axis=1, skipna=False)
    pd.testing.assert_series_equal(ind.signal(close, "ens_diverse_3"), expected, check_names=False)
    assert set(ind.ENSEMBLES["ens_all_trend_9"]) == set(ind.SIGNALS) - {"rsi14_reversal_30_70", "bb_reversion_20_2"}
