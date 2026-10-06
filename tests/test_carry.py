import numpy as np
import pandas as pd
import pytest

from alpha.strategies import carry

COINS = [f"C{i:02d}USDT" for i in range(20)]
END = pd.Timestamp("2026-09-26", tz="UTC")  # 토요일 — 마지막 리밸런싱(일요일 봉)은 09-20


def make_closes(days: int = 200, up: tuple[str, ...] = ()) -> pd.DataFrame:
    """완만히 내리는 코인(60일 고점과 멀다)과 계속 오르는 코인(항상 60일 고점 = 스퀴즈 상태)."""
    index = pd.date_range(end=END, periods=days, freq="1D")
    t = np.arange(days)[:, None]
    falling = 1 - 0.002 + 0.004 * np.sin(t + np.arange(len(COINS)))
    rising = 1 + 0.003 + 0.002 * np.sin(t)
    steps = np.where(np.isin(COINS, up), rising, falling)
    return pd.DataFrame(100 * np.cumprod(steps, axis=0), index=index, columns=COINS)


def make_fundings(closes: pd.DataFrame, rates: dict[str, float] | None = None) -> dict[str, pd.Series]:
    """코인 i의 펀딩 = i × 1e-5 (8시간마다). C19가 가장 높고 C00이 가장 낮다."""
    settle = pd.date_range(closes.index[0] - pd.Timedelta(days=10), closes.index[-1] + pd.Timedelta(days=1), freq="8h")
    rates = rates or {c: i * 1e-5 for i, c in enumerate(COINS)}
    return {c: pd.Series(r, index=settle) for c, r in rates.items()}


def test_signal_sums_seven_days_before_bar_close_excluding_close_settlement():
    closes = make_closes()
    fundings = make_fundings(closes, {c: 1e-4 for c in COINS})
    t = closes.index[-1]
    close_time = t + pd.Timedelta(days=1)
    spiked = fundings["C00USDT"].copy()
    spiked[close_time] = 0.05  # 봉 마감 시각 정산은 다음 봉 몫
    spiked[close_time - pd.Timedelta(days=7)] = 0.01  # 창 시작 시각은 포함
    S = carry.funding_signal(closes, {"C00USDT": spiked.sort_index(), "C01USDT": fundings["C01USDT"]})
    assert S.loc[t, "C01USDT"] == pytest.approx(21 * 1e-4)
    assert S.loc[t, "C00USDT"] == pytest.approx(20 * 1e-4 + 0.01)
    gap = fundings["C01USDT"].drop(fundings["C01USDT"].index[-6:-3])  # 창 안 기록 24시간 공백
    assert np.isnan(carry.funding_signal(closes, {"C01USDT": gap}).loc[t, "C01USDT"])


def test_shorts_high_funding_longs_low_funding_weekly():
    closes = make_closes()
    W = carry.carry_weights(closes, make_fundings(closes))["W"]
    last = W.iloc[-1]
    assert set(last[last < 0].index) == {"C16USDT", "C17USDT", "C18USDT", "C19USDT"}  # 20개 × 20% = 4
    assert set(last[last > 0].index) == {"C00USDT", "C01USDT", "C02USDT", "C03USDT"}
    assert last[last > 0].sum() == pytest.approx(0.5) and last[last < 0].sum() == pytest.approx(-0.5)
    changed = W.index[W.diff().abs().sum(axis=1) > 0]
    assert (changed.dayofweek == 6).all()  # 비중은 일요일 봉에서만 바뀐다


def test_squeeze_guard_skips_shorts_at_60_day_high_and_fills_next():
    closes = make_closes(up=("C19USDT",))
    fundings = make_fundings(closes)
    base = carry.carry_weights(closes, fundings)["W"].iloc[-1]
    guarded = carry.carry_weights(closes, fundings, squeeze_guard=True)["W"].iloc[-1]
    assert base["C19USDT"] < 0 and guarded["C19USDT"] == 0
    assert set(guarded[guarded < 0].index) == {"C15USDT", "C16USDT", "C17USDT", "C18USDT"}
    assert guarded[guarded > 0].to_dict() == pytest.approx(base[base > 0].to_dict())  # 롱 다리는 그대로


def test_squeeze_guard_exits_held_short_midweek_until_next_rebalance():
    closes = make_closes()
    wednesday = pd.Timestamp("2026-09-23", tz="UTC")
    closes.loc[wednesday:, "C18USDT"] *= 1.5  # 보유 숏이 주중에 60일 고점으로
    W = carry.carry_weights(closes, make_fundings(closes), squeeze_guard=True)["W"]["C18USDT"]
    assert W.loc[:wednesday - pd.Timedelta(days=1)].iloc[-1] < 0
    assert (W.loc[wednesday:] == 0).all()


def test_short_live_history_gives_same_last_weights_as_full_history():
    rng = np.random.default_rng(1)
    index = pd.date_range(end=END, periods=500, freq="1D")
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.03, (500, len(COINS))), axis=0), index=index, columns=COINS)
    settle = pd.date_range(index[0], END + pd.Timedelta(days=1), freq="8h")
    fundings = {c: pd.Series(rng.normal(1e-4, 3e-4, len(settle)), index=settle) for c in COINS}
    full = carry.carry_weights(closes, fundings, squeeze_guard=True)["W"]
    for end in index[-14:]:
        close_time = end + pd.Timedelta(days=1)
        recent = {c: f[(f.index >= close_time - pd.Timedelta(days=20)) & (f.index < close_time)] for c, f in fundings.items()}
        short = carry.carry_weights(closes.loc[:end].iloc[-300:], recent, squeeze_guard=True)["W"].iloc[-1]
        assert short.to_numpy() == pytest.approx(full.loc[end].to_numpy(), abs=1e-12)
    assert full.iloc[-1].abs().sum() > 0
