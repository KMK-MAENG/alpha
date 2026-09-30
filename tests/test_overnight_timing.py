import numpy as np
import pandas as pd
import pytest

from alpha.mining import overnight_timing as ot


def synthetic_daily(n=600, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=n, name="date")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    qqq = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": rng.uniform(1e6, 2e6, n),
        },
        index=index,
    )
    vix = pd.DataFrame({"close": 20 + rng.normal(0, 2, n)}, index=index)
    return qqq, vix, index.append(pd.bdate_range(index[-1] + pd.offsets.BDay(), periods=10))


def test_features_do_not_look_ahead():
    qqq, vix, sessions = synthetic_daily()
    full = ot.features(qqq, vix, sessions).iloc[:400]
    head = ot.features(qqq.iloc[:400], vix.iloc[:400], sessions)
    pd.testing.assert_frame_equal(full, head)


def test_calendar_features():
    qqq, vix, sessions = synthetic_daily()
    X = ot.features(qqq, vix, sessions)
    friday = X.index[X.index.dayofweek == 4][0]
    assert X.loc[friday, "days_to_next_session"] == 3
    assert X.loc[friday, "fri"] == 1.0


def test_target_is_next_overnight_return():
    qqq, _, _ = synthetic_daily()
    y = ot.target(qqq)
    assert y.iloc[0] == pytest.approx(qqq["open"].iloc[1] / qqq["close"].iloc[0] - 1)
    assert np.isnan(y.iloc[-1])


def test_screen_and_select_keep_only_planted_predictor():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("1999-01-01", "2026-09-30")
    X = pd.DataFrame(rng.normal(size=(len(index), 5)), index=index, columns=[f"x{i}" for i in range(5)])
    y = 0.0005 + 0.001 * X["x0"] + pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    assert ot.select(ot.screen(X, y)) == ["x0"]


def test_evaluate_charges_cost_only_on_held_nights():
    index = pd.bdate_range("2020-01-01", periods=4)
    y = pd.Series([0.002, -0.001, 0.003, 0.0], index=index)
    hold = pd.Series([True, False, True, False], index=index)
    pnl = ot.nightly_pnl(hold, y)
    assert pnl.tolist() == pytest.approx([0.002 - ot.COST, 0.0, 0.003 - ot.COST, 0.0])


def test_standardize_uses_only_past_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 3)), index=pd.bdate_range("2019-01-02", periods=600), columns=list("abc"))
    pd.testing.assert_frame_equal(ot.standardize(X).iloc[:400], ot.standardize(X.iloc[:400]))


def test_standardize_keeps_discrete_features_raw():
    index = pd.bdate_range("2019-01-02", periods=300)
    X = pd.DataFrame({"fri": (index.dayofweek == 4).astype(float), "ret_1d": np.linspace(-1, 1, 300)}, index=index)
    z = ot.standardize(X)
    pd.testing.assert_series_equal(z["fri"], X["fri"])
    assert z["ret_1d"].iloc[-1] != X["ret_1d"].iloc[-1]


def test_gp_finds_planted_night_rule():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("1999-01-01", "2026-09-30")
    X = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index, columns=[f"x{i}" for i in range(4)])
    y = 0.0005 + 0.01 * (X["x0"] < -1) + pd.Series(rng.normal(0, 0.008, len(index)), index=index)
    table, hold = ot.gp_search(X, y, pop_size=200, generations=3, seed=0)
    assert "x0" in table.index[0]
    assert table.loc["combo", "test"] > 1.0
