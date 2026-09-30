import numpy as np
import pandas as pd

from alpha.mining import combine, search


def noisy_predictors(n_signals=10, n=60_000, noise=30.0, seed=0):
    """다음 봉 수익률에 큰 잡음을 섞은 약한 신호들 (융합 계산 검증용이라 미래값을 일부러 쓴다)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-09-09", periods=n, freq="1h", tz="UTC")
    ret_next = pd.Series(rng.normal(0, 0.01, n), index=index)
    signals = {f"s{i}": ret_next + pd.Series(rng.normal(0, 0.01 * noise, n), index=index) for i in range(n_signals)}
    return signals, ret_next


def gross_train_sharpe(pos, ret_next):
    return search.sharpe(search.period(pos * ret_next, "train"))


def test_orient_keeps_informative_signals_and_flips_direction():
    signals, ret_next = noisy_predictors()
    signals["flipped"] = -signals["s0"]
    positions = combine.orient(signals, ret_next)
    assert set(signals) == set(positions.columns)
    assert gross_train_sharpe(positions["flipped"], ret_next) > 0


def test_equal_fusion_beats_every_single_signal():
    signals, ret_next = noisy_predictors()
    positions = combine.orient(signals, ret_next)
    fused = search.to_position(combine.fuse(positions, ret_next, "equal"))
    best_single = max(gross_train_sharpe(positions[c], ret_next) for c in positions)
    assert gross_train_sharpe(fused, ret_next) > 1.5 * best_single


def test_smoothing_reduces_turnover():
    pos = pd.Series(np.random.default_rng(0).choice([-1.0, 1.0], 1_000))
    assert combine.smooth(pos, 24).diff().abs().mean() < pos.diff().abs().mean() / 5
    assert combine.smooth(pos, 0).equals(pos)
