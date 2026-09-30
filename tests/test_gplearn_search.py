import numpy as np

from alpha.mining import expr, gplearn_search
from test_mining import synthetic_candles


def test_ts_functions_match_expr_ops_with_nan_as_zero():
    x = synthetic_candles(500)["close"]
    by_name = {f.name: f for f in gplearn_search.FUNCTIONS if not isinstance(f, str)}
    expected = expr.compute(("ts_zscore", "close", 24), {"close": x}).fillna(0).to_numpy()
    np.testing.assert_allclose(by_name["ts_zscore_24"](x.to_numpy()), expected)


def test_search_finds_planted_mean_reversion():
    candles = synthetic_candles(n=60_000, reversion=-0.3)
    report = gplearn_search.run(candles, pop_size=200, generations=3, seed=0)
    alphas = report.drop(index=["combo", "buy_and_hold"], errors="ignore")
    assert len(alphas) > 0
    assert alphas["test"].iloc[0] > 1.0


def test_search_finds_signal_planted_only_in_derivative_feature():
    candles = synthetic_candles(n=60_000)
    ret_next = candles["close"].pct_change().shift(-1).fillna(0)
    noise = np.random.default_rng(1).normal(0, 0.01, len(candles))
    candles["funding_rate"] = ret_next + noise  # 새 피처에만 신호를 심는다 (배선 검증용 미래값)
    report = gplearn_search.run(candles, pop_size=200, generations=3, seed=0)
    alphas = report.drop(index=["combo", "buy_and_hold"], errors="ignore")
    assert "funding_rate" in alphas.index[0]
    assert alphas["test"].iloc[0] > 1.0
