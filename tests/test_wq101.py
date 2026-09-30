import numpy as np
import pandas as pd
import pytest

from alpha.mining import expr, gplearn_search, wq101
from test_mining import synthetic_candles


def test_translate_arithmetic_and_constants():
    assert wq101.translate("((close - open) / ((high - low) + .001))") == (
        "div", ("sub", "close", "open"), ("add", ("sub", "high", "low"), 0.001)
    )
    assert wq101.translate("(-1 * correlation(open, volume, 10))") == ("neg", ("ts_corr_12", "open", "volume"))
    assert wq101.translate("(0 - (1 * close))") == ("neg", "close")


def test_translate_cross_sectional_ops_and_inputs():
    assert wq101.translate("rank(returns)") == ("ts_rank_24", "ret")
    assert wq101.translate("scale(indneutralize(vwap, IndClass.sector))") == "vwap"
    assert wq101.translate("(volume / adv20)") == ("div", "volume", ("ts_mean_24", "volume"))


def test_translate_windows():
    assert wq101.translate("delta(close, 1)") == ("ts_delta_1", "close")
    assert wq101.translate("delay(close, 2.9)") == ("delay_2", "close")  # 소수 윈도우는 버림
    assert wq101.translate("sum(close, 1)") == "close"  # 윈도우 1이면 x 자체
    assert wq101.translate("stddev(close, 250)") == ("ts_std_168", "close")
    assert wq101.translate("(sum(close, 8) / 8)") == ("ts_mean_6", "close")
    assert wq101.translate("sum(returns, 250)") == ("mul", 250.0, ("ts_mean_168", "ret"))


def test_translate_conditionals_and_min_max():
    assert wq101.translate("((returns < 0) ? stddev(returns, 20) : close)") == (
        "where", ("lt", "ret", 0.0), ("ts_std_24", "ret"), "close"
    )
    assert wq101.translate("(high > low)") == ("lt", "low", "high")
    assert wq101.translate("((close < open) || (close == open))") == ("lt", "close", "open")
    assert wq101.translate("max(close, 5)") == ("ts_max_6", "close")
    assert wq101.translate("max(close, open)") == ("max", "close", "open")


def test_all_formulas_translate_except_unsupported():
    assert set(wq101.SEEDS) == set(range(1, 102)) - {56, 81}
    with pytest.raises(wq101.Unsupported):
        wq101.translate(wq101.FORMULAS[56])


def test_decay_linear_and_argmax():
    x = pd.Series([1.0, 2.0, 3.0, 1.0])
    decay = wq101.OPS["decay_linear"][0](x, 3)
    assert decay.iloc[2] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)
    assert wq101.OPS["ts_argmax"][0](x, 3).tolist()[2:] == [2.0, 1.0]


def test_functions_do_not_look_ahead():
    rng = np.random.default_rng(0)
    full = [rng.normal(size=500) for _ in range(3)]
    for f in wq101.FUNCTIONS:
        if isinstance(f, str):
            continue
        a = f(*full[: f.arity])[:300]
        b = f(*(x[:300] for x in full[: f.arity]))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, err_msg=f.name)


def test_seeds_execute_to_finite_signals():
    X = gplearn_search.inputs(synthetic_candles(2_000))[0]
    for n, e in wq101.SEEDS.items():
        signal = gplearn_search.execute(e, X, wq101.FUNCTIONS)
        assert len(signal) == len(X) and np.isfinite(signal).all(), n


def test_run_with_seeds_finds_planted_mean_reversion():
    candles = synthetic_candles(n=60_000, reversion=-0.3)
    report = gplearn_search.run(
        candles, pop_size=200, generations=2, seed=0, function_set=wq101.FUNCTIONS, seeds=wq101.SEEDS
    )
    alphas = report.drop(index=["combo", "buy_and_hold"], errors="ignore")
    assert len(alphas) > 0
    assert alphas["test"].iloc[0] > 1.0
