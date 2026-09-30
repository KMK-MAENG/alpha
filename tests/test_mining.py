import random

import numpy as np
import pandas as pd
import pytest

from alpha.mining import expr, search


def synthetic_candles(n=20_000, reversion=0.0, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    ret = np.empty(n)
    ret[0] = eps[0]
    for t in range(1, n):
        ret[t] = reversion * ret[t - 1] + eps[t]
    close = 100 * np.cumprod(1 + ret)
    volume = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {
            "open": close / (1 + ret),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(1000, 2000, n),
            "taker_buy_volume": volume * rng.uniform(0.3, 0.7, n),
            "taker_buy_quote_volume": volume * close * 0.5,
        },
        index=pd.date_range("2019-09-09", periods=n, freq="1h", tz="UTC", name="open_time"),
    )


def test_features_match_feature_names():
    assert list(expr.features(synthetic_candles(100))) == expr.FEATURES


def test_features_include_derivative_columns_when_present():
    candles = synthetic_candles(100).assign(funding_rate=0.0001)
    assert list(expr.features(candles)) == expr.FEATURES + ["funding_rate"]


def test_compute_ts_ops():
    feats = {"close": pd.Series([1.0, 2.0, 4.0, 7.0])}
    assert expr.compute(("ts_delta", "close", 1), feats).tolist()[1:] == [1.0, 2.0, 3.0]
    assert expr.compute(("ts_mean", "close", 2), feats).tolist()[1:] == [1.5, 3.0, 5.5]
    assert expr.to_str(("ts_mean", ("neg", "close"), 24)) == "ts_mean(neg(close), 24)"


def test_random_expressions_do_not_look_ahead():
    candles = synthetic_candles(1_000)
    full, head = expr.features(candles), expr.features(candles.iloc[:600])
    rng = random.Random(0)
    for _ in range(100):
        e = expr.random_expr(rng, expr.MAX_DEPTH)
        a = expr.compute(e, full).iloc[:600]
        b = expr.compute(e, head)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, err_msg=expr.to_str(e))


def test_mutate_and_crossover_respect_max_depth():
    rng = random.Random(0)
    pop = [expr.random_expr(rng, expr.MAX_DEPTH) for _ in range(50)]
    for _ in range(500):
        a, b = rng.sample(pop, 2)
        for child in (expr.mutate(rng, a), expr.crossover(rng, a, b)):
            assert expr.depth(child) <= expr.MAX_DEPTH


def test_pnl_charges_fee_on_position_change():
    pos = pd.Series([0.0, 1.0, 1.0, -1.0])
    ret_next = pd.Series([0.01, 0.02, 0.03, 0.04])
    expected = [0.0, 0.02 - search.FEE, 0.03, -0.04 - 2 * search.FEE]
    assert search.pnl(pos, ret_next).tolist() == pytest.approx(expected)


def test_search_finds_planted_mean_reversion():
    candles = synthetic_candles(n=60_000, reversion=-0.3)
    report = search.run(candles, pop_size=60, generations=5, seed=0)
    alphas = report.drop(index=["combo", "buy_and_hold"], errors="ignore")
    assert len(alphas) > 0
    assert alphas["test"].iloc[0] > 1.0



def test_set_interval_scales_annualization_and_z_window(monkeypatch):
    monkeypatch.setattr(search, "BARS_PER_YEAR", search.BARS_PER_YEAR)
    monkeypatch.setattr(search, "Z_WINDOW", search.Z_WINDOW)
    search.set_interval("1h")
    assert (search.BARS_PER_YEAR, search.Z_WINDOW) == (24 * 365, 720)
    search.set_interval("4h")
    assert (search.BARS_PER_YEAR, search.Z_WINDOW) == (6 * 365, 180)
