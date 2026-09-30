import numpy as np
import pandas as pd
import pytest

from alpha.backtest import momentum as bt
from alpha.strategies import momentum


def daily_close(values, unit="ms"):
    index = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC").as_unit(unit)
    return pd.Series(values, index=index, dtype="float64")


def test_signal_averages_signs_of_lookback_returns():
    close = daily_close([100, 110, 105, 120])
    s = momentum.signal(close, lookbacks=[1, 2])
    # t=2: 1일 수익률 -, 2일 수익률 + → 0 / t=3: 둘 다 + → 1
    assert s.tolist()[2:] == [0.0, 1.0]
    assert np.isnan(s.iloc[1])


def test_position_uses_only_past_prices():
    close = daily_close(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.03, 400)))
    full = momentum.position(close).iloc[:300]
    head = momentum.position(close.iloc[:300])
    pd.testing.assert_series_equal(full, head)


def test_position_scales_by_volatility_and_caps_leverage():
    rng = np.random.default_rng(0)
    calm = daily_close(100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.001, 300)))  # 꾸준한 상승, 낮은 변동성
    assert momentum.position(calm).iloc[-1] == momentum.MAX_LEVERAGE
    wild = daily_close(100 * np.cumprod(1 + 0.01 + rng.normal(0, 0.08, 300)))
    assert 0 < abs(momentum.position(wild).iloc[-1]) < 1


def test_long_only_never_shorts():
    close = daily_close(np.linspace(200, 100, 300))
    assert momentum.position(close, long_only=True).dropna().eq(0).all()
    assert (momentum.position(close).dropna() < 0).all()


def test_backtest_holds_previous_position_with_fee_and_funding():
    close = daily_close([100.0, 100.0, 110.0, 99.0])
    pos = pd.Series([0.0, 1.0, 1.0, 0.0], index=close.index)
    funding = pd.Series([0.001, 0.002], index=close.index[2:].as_unit("ns"))  # 단위가 달라도 반영돼야 한다
    pnl = bt.backtest(close, funding, pos)
    assert pnl["price"].tolist() == pytest.approx([0.0, 0.0, 0.10, -0.10])
    assert pnl["funding"].tolist() == pytest.approx([0.0, 0.0, -0.001, -0.002])
    assert pnl["fee"].tolist() == pytest.approx([0.0, 0.0, -bt.FEE, 0.0])


def test_portfolio_averages_components_over_available_coins():
    index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    a = pd.DataFrame({"price": [0.02, 0.02, 0.02], "total": [0.02, 0.02, 0.02]}, index=index)
    b = pd.DataFrame({"price": [0.04, 0.00], "total": [0.04, 0.00]}, index=index[1:])  # 하루 늦게 상장
    out = bt._portfolio({"A": {"s": a}, "B": {"s": b}}, "s")
    assert out["total"].tolist() == pytest.approx([0.02, 0.03, 0.01])


def trending_closes(n_coins=25, n_days=400, seed=0):
    """코인 i의 추세가 i에 비례하는 가격들 (i가 클수록 강한 모멘텀)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="1D", tz="UTC")
    drift = np.linspace(-0.004, 0.004, n_coins)
    rets = drift + rng.normal(0, 0.02, (n_days, n_coins))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=index, columns=[f"C{i:02d}" for i in range(n_coins)])


def test_cross_sectional_longs_winners_shorts_losers_with_unit_legs():
    weights = momentum.cross_sectional_weights(trending_closes())
    w = weights.iloc[-1]
    assert w[w > 0].sum() == pytest.approx(1.0) and w[w < 0].sum() == pytest.approx(-1.0)
    recent = weights.iloc[-100:]
    top, bottom = [f"C{i:02d}" for i in range(15, 25)], [f"C{i:02d}" for i in range(10)]
    assert recent[top].clip(lower=0).sum().sum() / recent.clip(lower=0).sum().sum() > 0.8  # 롱은 주로 강한 추세 코인
    assert recent[bottom].clip(upper=0).sum().sum() / recent.clip(upper=0).sum().sum() > 0.8  # 숏은 주로 약한 추세 코인


def test_cross_sectional_long_only_rebalances_only_on_thursdays():
    w = momentum.cross_sectional_weights(trending_closes(), long_only=True)
    assert (w >= 0).all().all()
    changes = w.diff().abs().sum(axis=1)
    # 데이터 시작일과 무관하게 달력 기준(1970-01-01 목요일부터 7일마다)으로 리밸런싱한다
    assert (changes[changes > 0].index.dayofweek == 3).all()


def test_cross_sectional_needs_minimum_coins_and_no_lookahead():
    closes = trending_closes()
    assert momentum.cross_sectional_weights(closes.iloc[:, :10]).abs().sum().sum() == 0  # 20개 미만이면 무포지션
    full = momentum.cross_sectional_weights(closes).iloc[:300]
    head = momentum.cross_sectional_weights(closes.iloc[:300])
    pd.testing.assert_frame_equal(full, head)


def test_portfolio_pnl_sums_coins_and_short_receives_positive_funding():
    closes = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 90.0, 90.0]},
                          index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"))
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=closes.index)
    funding = pd.Series([0.001], index=closes.index[2:])
    pnl = bt.portfolio_pnl(closes, {"A": funding, "B": funding}, weights)
    assert pnl["price"].tolist() == pytest.approx([0.0, 0.05 + 0.05, 0.0])  # A 롱 +10%, B 숏 -10%
    assert pnl["funding"].iloc[2] == pytest.approx(-0.5 * 0.001 + 0.5 * 0.001)  # 롱은 내고 숏은 받는다
    assert pnl["fee"].iloc[1] == pytest.approx(-bt.FEE * 1.0)


def test_dual_momentum_goes_to_cash_when_every_trend_turns_down():
    closes = trending_closes(n_days=500)
    # 마지막 150일 모든 코인이 매일 1% 안팎 하락 (순위가 갈리도록 코인마다 하락률을 조금씩 다르게)
    daily = 0.99 - 0.0002 * np.arange(closes.shape[1])
    closes.iloc[-150:] = closes.iloc[-151].to_numpy() * np.cumprod(np.tile(daily, (150, 1)), axis=0)
    assert momentum.cross_sectional_weights(closes, long_only=True).iloc[-1].sum() == pytest.approx(1.0)
    assert momentum.dual_momentum_weights(closes).iloc[-1].sum() == 0.0


def test_dual_momentum_keeps_only_uptrending_selected_coins():
    closes = trending_closes()
    dual = momentum.dual_momentum_weights(closes).iloc[-1]
    xs = momentum.cross_sectional_weights(closes, long_only=True).iloc[-1]
    assert (dual <= xs + 1e-12).all() and (dual[dual > 0] == xs[dual > 0]).all()
    assert (momentum.signal(closes[dual[dual > 0].index[0]]).iloc[-1]) > 0


def test_vol_targeted_hits_target_and_respects_cap_and_cash():
    closes = trending_closes(n_days=800)
    market = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.03, len(closes)))  # 코인 공통 움직임 (배율 상한에 안 걸리게)
    closes = closes.mul(market, axis=0)
    weights = momentum.cross_sectional_weights(closes, quantile=1.0, long_only=True)  # 전체 바스켓
    scaled = momentum.vol_targeted(weights, closes)
    realized = (scaled.shift(1) * closes.pct_change()).sum(axis=1).iloc[200:].std() * np.sqrt(365)
    assert realized == pytest.approx(momentum.TARGET_VOL, rel=0.15)
    scale = scaled.sum(axis=1) / weights.sum(axis=1)
    assert scale.dropna().max() <= momentum.MAX_LEVERAGE + 1e-12
    assert momentum.vol_targeted(weights * 0.0, closes).abs().sum().sum() == 0


def test_time_series_weights_average_positions_over_available_coins():
    closes = trending_closes()
    closes.iloc[:150, 0] = np.nan  # 늦게 상장한 코인
    w = momentum.time_series_weights(closes)
    positions = closes.apply(lambda c: momentum.position(c, long_only=True))
    expected = positions.iloc[-1].sum() / positions.iloc[-1].notna().sum()
    assert w.iloc[-1].sum() == pytest.approx(expected)
    assert w.iloc[160, 0] == 0.0 or positions.iloc[160, 0] > 0


def test_four_hour_bars_convert_days_to_bars():
    closes = trending_closes(n_days=2400)  # 봉 2,400개 = 4h 기준 400일
    closes.index = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    w = momentum.cross_sectional_weights(closes, long_only=True, bars_per_day=6)
    changed = w.index[w.diff().abs().sum(axis=1) > 0]
    assert len(changed) > 0 and (changed.dayofweek == 3).all() and (changed.hour == 0).all()

    market = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.012, len(closes)))
    closes = closes.mul(market, axis=0)
    basket = momentum.cross_sectional_weights(closes, quantile=1.0, long_only=True, bars_per_day=6)
    scaled = momentum.vol_targeted(basket, closes, bars_per_day=6)
    realized = (scaled.shift(1) * closes.pct_change()).sum(axis=1).iloc[1200:].std() * np.sqrt(365 * 6)
    assert realized == pytest.approx(momentum.TARGET_VOL, rel=0.15)


@pytest.mark.parametrize("options", [{}, momentum.LIVE_OPTIONS])
def test_combined_weights_today_do_not_depend_on_history_length(options):
    """실거래는 최근 300일만 받는다 — 오늘 비중이 전체 기록으로 계산한 백테스트와 같아야 한다 (진입 확인은 상태를 기억한다)."""
    closes = trending_closes(n_days=800)
    full = momentum.combined_weights(closes, **options).iloc[-1]
    recent = momentum.combined_weights(closes.iloc[-300:], **options).iloc[-1]
    assert full.abs().sum() > 0
    pd.testing.assert_series_equal(recent, full)


def test_combined_weights_is_half_dual_half_time_series():
    closes = trending_closes(n_days=500)
    dual = momentum.vol_targeted(momentum.dual_momentum_weights(closes), closes)
    ts = momentum.vol_targeted(momentum.time_series_weights(closes), closes)
    pd.testing.assert_frame_equal(momentum.combined_weights(closes), 0.5 * dual + 0.5 * ts)


def volumes_like(closes, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(1, 10, closes.shape[1])  # 코인 번호가 클수록 거래대금이 크다
    return pd.DataFrame(base * rng.uniform(0.8, 1.2, closes.shape), index=closes.index, columns=closes.columns)


def test_liquidity_universe_picks_top_volume_monthly_after_one_year():
    closes = trending_closes(n_coins=6, n_days=800)
    closes.iloc[:300, 5] = np.nan  # 가장 거래대금이 큰 코인이 늦게 상장
    u = momentum.liquidity_universe(closes, volumes_like(closes), top_n=2)
    changes = u.index[(u != u.shift()).any(axis=1)][1:]
    assert (changes.day == 1).all()  # 목록은 매달 첫 봉에만 바뀐다
    assert not u.iloc[:365].any().any()  # 상장 1년이 안 된 코인뿐인 기간엔 비어 있다
    late_member_from = u.index[u["C05"]][0]
    assert late_member_from >= closes.index[300] + pd.Timedelta(days=365)
    assert set(u.columns[u.iloc[-1]]) == {"C04", "C05"}
    full = momentum.liquidity_universe(closes, volumes_like(closes), top_n=2).iloc[:600]
    head = momentum.liquidity_universe(closes.iloc[:600], volumes_like(closes).iloc[:600], top_n=2)
    pd.testing.assert_frame_equal(full, head)


def test_universe_restricts_ranking_and_averaging_to_members():
    closes = trending_closes(n_days=500)
    everyone = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    pd.testing.assert_frame_equal(momentum.combined_weights(closes, universe=everyone), momentum.combined_weights(closes))

    members = everyone.copy()
    members[["C24", "C23"]] = False  # 가장 강한 추세 코인 둘을 목록에서 뺀다
    xs = momentum.cross_sectional_weights(closes, long_only=True, universe=members)
    ts = momentum.time_series_weights(closes, universe=members)
    assert (xs[["C24", "C23"]] == 0).all().all() and (ts[["C24", "C23"]] == 0).all().all()
    assert xs.iloc[-1].sum() == pytest.approx(1.0)  # 남은 23개 안에서 상위 20%를 다시 고른다
    positions = closes.apply(lambda c: momentum.position(c, long_only=True)).iloc[-1].drop(["C24", "C23"])
    assert ts.iloc[-1].sum() == pytest.approx(positions.sum() / positions.notna().sum())


def test_portfolio_pnl_ignores_price_gaps():
    """상장 폐지 후 재상장처럼 가격 공백이 있으면 공백 전후 가격 차이를 수익률로 잡지 않는다."""
    index = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    closes = pd.DataFrame({"A": [100.0, 100.0, np.nan, np.nan, 5.0]}, index=index)  # 액면 변경 후 재상장
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=index)
    pnl = bt.portfolio_pnl(closes, {"A": pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))}, weights)
    assert pnl["price"].abs().max() == 0.0


def test_confirmed_entry_needs_all_three_up_but_holds_while_two_up():
    s = pd.Series([1 / 3, 1.0, 1 / 3, -1 / 3, 1 / 3, 1.0, np.nan])
    assert momentum.confirmed(s).tolist()[:6] == pytest.approx([0.0, 1.0, 1 / 3, 0.0, 0.0, 1.0])
    assert np.isnan(momentum.confirmed(s).iloc[-1])


def test_downside_volatility_keeps_size_in_upside_volatile_rally():
    rng = np.random.default_rng(0)
    rally = daily_close(100 * np.cumprod(1 + np.abs(rng.normal(0.01, 0.05, 300))))  # 크게 흔들리며 오르기만 함
    assert momentum.position(rally, long_only=True, downside=True).iloc[-1] == momentum.MAX_LEVERAGE
    assert momentum.position(rally, long_only=True).iloc[-1] < 1.0


def test_rank_buffer_keeps_winners_until_outside_exit_quantile():
    closes = trending_closes()
    plain = momentum.cross_sectional_weights(closes, long_only=True)
    buffered = momentum.cross_sectional_weights(closes, long_only=True, exit_quantile=0.4)
    turnover = lambda w: w.diff().abs().sum().sum()
    assert turnover(buffered) < turnover(plain)
    assert buffered.iloc[-1].sum() == pytest.approx(1.0)
    assert ((buffered > 0).sum(axis=1) >= (plain > 0).sum(axis=1)).loc[plain.abs().sum(axis=1) > 0].all()


def test_trend_options_default_to_current_behavior():
    closes = trending_closes(n_days=500)
    pd.testing.assert_frame_equal(
        momentum.combined_weights(closes),
        momentum.combined_weights(closes, downside=False, confirm_entry=False, exit_quantile=None),
    )
    assert not momentum.combined_weights(closes, downside=True, confirm_entry=True, exit_quantile=0.4).equals(
        momentum.combined_weights(closes)
    )


def test_confirmed_short_side_mirrors_long_side():
    s = pd.Series([-1 / 3, -1.0, -1 / 3, 1 / 3, -1 / 3, -1.0, 1.0])
    # 숏: 세 기간 모두 마이너스면 진입, 두 기간 마이너스면 1/3, 플러스로 돌면 청산. -1 → +1이면 바로 롱으로 뒤집힌다
    assert momentum.confirmed(s, allow_short=True).tolist() == pytest.approx([0, -1, -1 / 3, 0, 0, -1, 1])
    assert momentum.confirmed(s).tolist() == pytest.approx([0, 0, 0, 0, 0, 0, 1])  # 기본은 롱 전용


def test_short_options_add_shorts_only_where_downtrend_is_confirmed():
    closes = trending_closes(n_days=500)
    base = momentum.combined_weights(closes, confirm_entry=True)
    pd.testing.assert_frame_equal(base, momentum.combined_weights(closes, confirm_entry=True, allow_short=False))
    assert (base >= 0).all().all()
    both = momentum.dual_momentum_weights(closes, confirm_entry=True, allow_short=True)
    recent = both.iloc[-100:].clip(upper=0)
    weak = [f"C{i:02d}" for i in range(10)]
    assert recent.sum().sum() < 0 and recent[weak].sum().sum() / recent.sum().sum() > 0.8  # 숏은 주로 약한 추세 코인
    ts = momentum.time_series_weights(closes, confirm_entry=True, allow_short=True)
    assert (ts.iloc[-1] < 0).sum() > 0 and (ts.iloc[-1] > 0).sum() > 0


def test_market_filter_blocks_shorts_while_market_trends_up():
    closes = trending_closes(n_days=500)
    closes["BTCUSDT"] = np.linspace(100, 300, len(closes))  # 시장(BTC)은 계속 상승
    w = momentum.combined_weights(closes, confirm_entry=True, allow_short=True, short_market="BTCUSDT")
    assert (w.iloc[-200:] >= 0).all().all()


def test_ts_short_adds_shorts_only_to_time_series_engine():
    closes = trending_closes(n_days=500)
    w = momentum.combined_weights(closes, confirm_entry=True, ts_short=True)
    dual = momentum.vol_targeted(momentum.dual_momentum_weights(closes, confirm_entry=True), closes)
    ts = momentum.vol_targeted(momentum.time_series_weights(closes, confirm_entry=True, allow_short=True), closes)
    pd.testing.assert_frame_equal(w, 0.5 * dual + 0.5 * ts)
    assert (dual >= 0).all().all() and (w < 0).any().any()
