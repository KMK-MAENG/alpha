"""모멘텀 신호.

- 시계열(position): 룩백별 과거 수익률 부호의 평균(-1 ~ +1) × min(목표 변동성 / 최근 변동성, 최대 레버리지)
- 횡단면(cross_sectional_weights): 룩백별 수익률의 코인 간 백분위 순위를 평균한 점수로 상위 QUANTILE 롱,
  하위 QUANTILE 숏. 각 쪽 안에서 변동성 역비중, 롱 합 +1 / 숏 합 -1, 달력 기준 REBALANCE_DAYS마다 리밸런싱
  (1970-01-01부터 7일마다 = 매주 목요일 봉. 데이터 시작일과 무관해야 실거래와 백테스트가 같은 날 리밸런싱한다)
- 듀얼(dual_momentum_weights): 횡단면 롱 전용 비중 중 그 코인의 시계열 신호가 양수인 것만 보유 (나머지는 현금).
  필터는 매일 적용한다
- time_series_weights: 시계열 롱/무포지션 포지션을 거래 가능 코인 동일 비중 포트폴리오 비중으로
- vol_targeted: 비중 전체를 포트폴리오 목표 변동성에 맞춰 조절
- combined_weights: 실거래 전략. 듀얼·시계열을 각각 변동성 타게팅한 뒤 자본 50:50
- 추세추종 보강 옵션 (기본값은 지금 동작, 실거래는 기본값을 쓴다):
  downside — 비중 조절에 전체 변동성 대신 하방 편차 × √2 (급등 변동성에는 비중을 줄이지 않는다)
  confirm_entry — 무포지션에서는 세 기간 모두 플러스일 때만 진입, 보유 중에는 기존 규칙 (휩쏘 감소)
  exit_quantile — 듀얼 순위 완충: 상위 quantile에 들면 매수, 리밸런싱 때 상위 exit_quantile 밖일 때만 매도
  allow_short — 두 엔진 모두 숏 추가 (진입 확인을 대칭으로: 세 기간 모두 마이너스면 숏 진입). 듀얼은 하위 quantile 중 하락 추세 확인 코인
  ts_short — 시계열 엔진에만 숏 추가 (듀얼은 롱 전용). 롱/숏 비율은 코인별 추세 신호가 정한다
  short_market — 이 코인(예: BTCUSDT) 자체가 하락 추세일 때만 숏 허용 (시장 필터)
- liquidity_universe: 매달 거래대금 상위 코인으로 목록을 다시 정한다. 비중 함수에 universe로 주면 순위·동일 비중 계산과
  보유를 그 목록 안에서만 한다 (주지 않으면 closes의 모든 코인)
날짜 t의 값은 t 종가까지의 정보만 쓴다. 기본 파라미터는 문헌의 표준 조합이며 이 데이터로 고른 값이 아니다.
기간(룩백·변동성 윈도우·리밸런싱)은 일 단위이고, bars_per_day(일봉 1, 4h 6)로 봉 개수로 환산한다.
"""
import numpy as np
import pandas as pd

LOOKBACKS = [20, 60, 120]  # 약 1·3·6개월 (일)
TARGET_VOL = 0.30  # 연환산
VOL_WINDOW = 60  # 일
MAX_LEVERAGE = 2.0
QUANTILE = 0.2
REBALANCE_DAYS = 7
MIN_COINS = 20  # 자격(신호·변동성 계산 가능) 코인이 이보다 적은 날은 무포지션
# 실거래 설정: 결합 50:50 + 진입 확인 + 시계열 엔진 숏
# (2026-09-30 --trend-variants로 진입 확인, --shorts로 시계열 숏 채택. 롱/숏 비율은 신호에 맡긴다)
LIVE_OPTIONS = {"confirm_entry": True, "ts_short": True}
# 결과를 보기 전에 정한 규칙: 바이낸스 USDT 무기한 선물 중 2020년 말 이전 상장, 현재 거래 중, 스테이블코인·지수 제외.
# 그 사이 상장 폐지된 코인은 데이터가 없어 빠지므로 생존 편향이 있다
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BCHUSDT", "XRPUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", "LINKUSDT", "XLMUSDT", "ADAUSDT",
    "XMRUSDT", "DASHUSDT", "ZECUSDT", "XTZUSDT", "ATOMUSDT", "BNBUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT", "VETUSDT",
    "NEOUSDT", "QTUMUSDT", "IOSTUSDT", "THETAUSDT", "ALGOUSDT", "ZILUSDT", "KNCUSDT", "ZRXUSDT", "COMPUSDT",
    "DOGEUSDT", "KAVAUSDT", "BANDUSDT", "RLCUSDT", "SNXUSDT", "DOTUSDT", "SKLUSDT", "YFIUSDT", "CRVUSDT", "TRBUSDT",
    "SUSHIUSDT", "RUNEUSDT", "EGLDUSDT", "SOLUSDT", "UNIUSDT", "AVAXUSDT", "ENJUSDT", "KSMUSDT", "NEARUSDT",
    "FILUSDT", "AAVEUSDT", "RSRUSDT", "BELUSDT", "AXSUSDT", "ZENUSDT", "GRTUSDT", "1INCHUSDT",
]


def _bars(days: float, bars_per_day: int) -> int:
    return max(1, round(days * bars_per_day))


def signal(close: pd.Series, lookbacks: list[int] = LOOKBACKS) -> pd.Series:
    """룩백(봉 개수)별 과거 수익률 부호의 평균."""
    return pd.concat([np.sign(close.pct_change(n)) for n in lookbacks], axis=1).mean(axis=1, skipna=False)


def confirmed(s: pd.Series, allow_short: bool = False) -> pd.Series:
    """진입 확인: 무포지션에서는 신호가 +1(세 기간 모두 플러스)일 때만 롱 진입하고, 보유 중에는 max(신호, 0)을 따른다
    (두 기간 플러스면 1/3, 두 기간 마이너스면 청산). allow_short면 숏도 거울상으로 (-1이면 진입, 플러스로 돌면 청산)."""
    out, state = np.full(len(s), np.nan), 0  # +1 롱 보유, -1 숏 보유, 0 무포지션
    for t, value in enumerate(s.to_numpy()):
        if np.isnan(value):
            continue
        if (state > 0 and value <= 0) or (state < 0 and value >= 0):
            state = 0
        if state == 0:
            state = 1 if value >= 1 else (-1 if allow_short and value <= -1 else 0)
        out[t] = max(value, 0.0) if state > 0 else (min(value, 0.0) if state < 0 else 0.0)
    return pd.Series(out, index=s.index)


def _downside_vol(returns: np.ndarray | pd.Series, window: int | None = None):
    """하방 편차 × √2 (대칭 분포면 표준편차와 같아지도록). window가 있으면 이동 계산."""
    if window is None:
        return np.sqrt(np.mean(np.minimum(returns, 0) ** 2) * 2)
    return (returns.clip(upper=0) ** 2).rolling(window).mean().mul(2) ** 0.5


def position(
    close: pd.Series,
    lookbacks: list[float] = LOOKBACKS,
    long_only: bool = False,
    bars_per_day: int = 1,
    downside: bool = False,
    confirm_entry: bool = False,
) -> pd.Series:
    s = signal(close, [_bars(n, bars_per_day) for n in lookbacks])
    if long_only:
        s = confirmed(s) if confirm_entry else s.clip(lower=0)
    elif confirm_entry:
        s = confirmed(s, allow_short=True)
    window = _bars(VOL_WINDOW, bars_per_day)
    r = close.pct_change()
    vol = (_downside_vol(r, window) if downside else r.rolling(window).std()) * np.sqrt(365 * bars_per_day)
    return s * (TARGET_VOL / vol).clip(upper=MAX_LEVERAGE)


def liquidity_universe(
    closes: pd.DataFrame,
    quote_volume: pd.DataFrame,
    top_n: int = 50,
    min_history_days: int = 365,
    volume_days: int = 30,
    bars_per_day: int = 1,
) -> pd.DataFrame:
    """봉 × 코인 목록 여부. 매달 첫 봉 종가 기준, 첫 봉 이후 min_history_days가 지난 코인 중 최근 volume_days
    평균 거래대금 상위 top_n을 고르고 그 달 동안 유지한다. 그 시점까지의 데이터만 쓴다."""
    first = pd.DatetimeIndex(closes.apply(pd.Series.first_valid_index))
    eligible_from = (first + pd.Timedelta(days=min_history_days)).as_unit("ns").asi8
    old_enough = closes.index.as_unit("ns").asi8[:, None] >= eligible_from[None, :]
    volume = quote_volume.reindex_like(closes).rolling(_bars(volume_days, bars_per_day), min_periods=1).mean()
    score = volume.where(old_enough & closes.notna())
    chosen = score.rank(axis=1, ascending=False, method="first") <= top_n
    month = pd.Series(closes.index.year * 12 + closes.index.month, index=closes.index)
    month_start = month != month.shift()
    return chosen.where(month_start, axis=0).ffill().astype(bool)


def cross_sectional_weights(
    closes: pd.DataFrame,
    lookbacks: list[float] = LOOKBACKS,
    quantile: float = QUANTILE,
    long_only: bool = False,
    bars_per_day: int = 1,
    universe: pd.DataFrame | None = None,
    exit_quantile: float | None = None,
) -> pd.DataFrame:
    """closes: 봉 × 코인 종가. 반환: 봉 × 코인 비중 (그 봉 종가에 정해 다음 봉부터 보유)."""
    vol = closes.pct_change().rolling(_bars(VOL_WINDOW, bars_per_day)).std()
    returns = [closes.pct_change(_bars(n, bars_per_day)) for n in lookbacks]
    if universe is not None:
        returns = [r.where(universe) for r in returns]
    ranks = [r.rank(axis=1, pct=True) for r in returns]
    score = (sum(ranks) / len(ranks)).where(vol.notna())
    pct = score.rank(axis=1, pct=True)

    inverse_vol = 1 / vol
    # 달력 기준 REBALANCE_DAYS마다(1970-01-01부터 센 봉 번호) 정한 비중을 다음 리밸런싱까지 유지한다
    bar_number = closes.index.as_unit("ns").asi8 // (86_400 * 10**9 // bars_per_day)
    rebalance = bar_number % _bars(REBALANCE_DAYS, bars_per_day) == 0

    def leg(enter: pd.DataFrame, stay: pd.DataFrame) -> pd.DataFrame:
        mask = _buffered(enter, stay, rebalance) if exit_quantile is not None else enter
        w = inverse_vol.where(mask)
        return w.div(w.sum(axis=1), axis=0).fillna(0.0)

    exit_q = exit_quantile or quantile
    weights = leg(pct > 1 - quantile, pct > 1 - exit_q)
    if not long_only:
        weights -= leg(pct <= quantile, pct <= exit_q)
    weights[score.notna().sum(axis=1) < MIN_COINS] = 0.0
    # 첫 리밸런싱 전에는 정한 비중이 없으므로 무포지션
    return weights.where(pd.Series(rebalance, index=weights.index), axis=0).ffill().fillna(0.0)


def _buffered(enter: pd.DataFrame, stay: pd.DataFrame, rebalance: np.ndarray) -> pd.DataFrame:
    """리밸런싱 봉마다: 새로 들어오려면 enter, 이미 보유 중이면 stay만 만족하면 유지."""
    out, keep = enter.to_numpy().copy(), stay.to_numpy()
    held = np.zeros(out.shape[1], dtype=bool)
    for t in np.flatnonzero(rebalance):
        held = out[t] | (held & keep[t])
        out[t] = held
    return pd.DataFrame(out, index=enter.index, columns=enter.columns)


def dual_momentum_weights(
    closes: pd.DataFrame,
    lookbacks: list[float] = LOOKBACKS,
    quantile: float = QUANTILE,
    bars_per_day: int = 1,
    universe: pd.DataFrame | None = None,
    confirm_entry: bool = False,
    exit_quantile: float | None = None,
    allow_short: bool = False,
) -> pd.DataFrame:
    selected = cross_sectional_weights(
        closes, lookbacks, quantile, long_only=not allow_short, bars_per_day=bars_per_day, universe=universe,
        exit_quantile=exit_quantile,
    )
    bars = [_bars(n, bars_per_day) for n in lookbacks]
    trend = closes.apply(lambda close: signal(close, bars))
    if confirm_entry:
        trend = trend.apply(lambda t: confirmed(t, allow_short))
    longs = selected.clip(lower=0).where(trend > 0, 0.0)
    if not allow_short:
        return longs
    return longs + selected.clip(upper=0).where(trend < 0, 0.0)


def time_series_weights(
    closes: pd.DataFrame,
    lookbacks: list[float] = LOOKBACKS,
    bars_per_day: int = 1,
    universe: pd.DataFrame | None = None,
    downside: bool = False,
    confirm_entry: bool = False,
    allow_short: bool = False,
) -> pd.DataFrame:
    """코인별 시계열 포지션(기본 롱/무포지션)을 그 봉에 신호가 있는 코인 수로 나눈 비중 (동일 비중 포트폴리오)."""
    positions = closes.apply(
        lambda close: position(
            close, lookbacks, not allow_short, bars_per_day, downside=downside, confirm_entry=confirm_entry
        )
    )
    if universe is not None:
        positions = positions.where(universe)
    return positions.fillna(0.0).div(positions.notna().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def vol_targeted(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    target_vol: float = TARGET_VOL,
    bars_per_day: int = 1,
    downside: bool = False,
) -> pd.DataFrame:
    """비중 전체에 배율을 곱해 포트폴리오 변동성을 target_vol에 맞춘다 (배율은 MAX_LEVERAGE 이하).
    변동성은 '현재 비중'을 최근 VOL_WINDOW일 코인 수익률에 적용해 추정한다. 전략 자신의 과거 수익률로 추정하면
    현금 구간 직후 변동성을 과소추정해 재진입 때 레버리지가 과해진다."""
    returns = closes.pct_change().fillna(0.0).to_numpy()
    w = weights.fillna(0.0).to_numpy()
    window = _bars(VOL_WINDOW, bars_per_day)
    scale = np.zeros(len(w))
    for t in range(window, len(w)):
        if not w[t].any():
            continue
        portfolio = returns[t - window + 1 : t + 1] @ w[t]
        vol = (_downside_vol(portfolio) if downside else portfolio.std(ddof=1)) * np.sqrt(365 * bars_per_day)
        scale[t] = min(target_vol / vol, MAX_LEVERAGE) if vol > 0 else 0.0
    return weights.fillna(0.0).mul(scale, axis=0)


def combined_weights(
    closes: pd.DataFrame,
    lookbacks: list[float] = LOOKBACKS,
    bars_per_day: int = 1,
    universe: pd.DataFrame | None = None,
    downside: bool = False,
    confirm_entry: bool = False,
    exit_quantile: float | None = None,
    allow_short: bool = False,
    short_market: str | None = None,
    ts_short: bool = False,
) -> pd.DataFrame:
    dual = dual_momentum_weights(
        closes, lookbacks, bars_per_day=bars_per_day, universe=universe, confirm_entry=confirm_entry,
        exit_quantile=exit_quantile, allow_short=allow_short,
    )
    ts = time_series_weights(
        closes, lookbacks, bars_per_day=bars_per_day, universe=universe, downside=downside,
        confirm_entry=confirm_entry, allow_short=allow_short or ts_short,
    )
    if short_market is not None:  # 시장 코인 자체가 하락 추세일 때만 숏
        market = signal(closes[short_market], [_bars(n, bars_per_day) for n in lookbacks])
        if confirm_entry:
            market = confirmed(market, allow_short=True)
        blocked = ~(market < 0).to_numpy()[:, None]
        dual, ts = dual.mask(dual.lt(0) & blocked, 0.0), ts.mask(ts.lt(0) & blocked, 0.0)
    dual = vol_targeted(dual, closes, bars_per_day=bars_per_day, downside=downside)
    ts = vol_targeted(ts, closes, bars_per_day=bars_per_day, downside=downside)
    return 0.5 * dual + 0.5 * ts
