"""시계열 모멘텀의 대안 신호 (지표 기반). 모두 -1 ~ +1이고 날짜 t 종가까지의 정보만 쓴다.

파라미터는 각 지표의 교과서 기본값이다 (이 데이터로 고른 값이 아님). 포지션 크기는 momentum.position과 같은
변동성 타게팅을 쓴다 — 신호만 바꿔 비교하기 위함. 돈치안 채널은 종가 기준이다 (고가·저가 대신).
"""
import numpy as np
import pandas as pd

from alpha.strategies import momentum


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """와일더 RSI (평균 상승폭·하락폭을 1/n 지수 평활)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)  # loss가 0이면 100


def rsi_reversal_positions(r: pd.Series, low: float = 30, high: float = 70, exit_level: float = 50) -> pd.Series:
    """역추세: RSI < low면 롱, RSI > high면 숏으로 들어가고 RSI가 exit_level을 넘어서면 청산."""
    out, pos = np.full(len(r), np.nan), 0.0
    for t, value in enumerate(r.to_numpy()):
        if np.isnan(value):
            continue
        if value < low:
            pos = 1.0
        elif value > high:
            pos = -1.0
        elif (pos > 0 and value > exit_level) or (pos < 0 and value < exit_level):
            pos = 0.0
        out[t] = pos
    return pd.Series(out, index=r.index)


def donchian(close: pd.Series, entry: int = 55, exit: int = 20) -> pd.Series:
    """터틀: 직전 entry일 최고가 돌파 시 롱(최저가 이탈 시 숏), 직전 exit일 반대쪽 채널을 깨면 청산."""
    hi_in, lo_in = close.rolling(entry).max().shift(1), close.rolling(entry).min().shift(1)
    hi_out, lo_out = close.rolling(exit).max().shift(1), close.rolling(exit).min().shift(1)
    out, pos = np.full(len(close), np.nan), 0.0
    for t, (c, hi, lo, ho, lo2) in enumerate(zip(close, hi_in, lo_in, hi_out, lo_out)):
        if np.isnan(hi):
            continue
        if c > hi:
            pos = 1.0
        elif c < lo:
            pos = -1.0
        elif (pos > 0 and c < lo2) or (pos < 0 and c > ho):
            pos = 0.0
        out[t] = pos
    return pd.Series(out, index=close.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(중심선, 상단, 하단). 표준편차는 볼린저 정의대로 모집단 표준편차."""
    mid, sd = sma(close, n), close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def band_positions(close: pd.Series, breakout: bool, n: int = 20, k: float = 2.0) -> pd.Series:
    """breakout: 상단 돌파 롱 / 하단 이탈 숏, 중심선을 되넘으면 청산.
    reversion(breakout=False): 하단 이탈 롱 / 상단 돌파 숏, 중심선을 회복하면 청산."""
    mid, upper, lower = bollinger(close, n, k)
    out, pos = np.full(len(close), np.nan), 0.0
    for t, (c, m, u, lo) in enumerate(zip(close, mid, upper, lower)):
        if np.isnan(m):
            continue
        if c > u:
            pos = 1.0 if breakout else -1.0
        elif c < lo:
            pos = -1.0 if breakout else 1.0
        elif (pos > 0 and (c < m if breakout else c > m)) or (pos < 0 and (c > m if breakout else c < m)):
            pos = 0.0
        out[t] = pos
    return pd.Series(out, index=close.index)


def _percent_b(close: pd.Series) -> pd.Series:
    """밴드 안 위치: (종가 - 중심선) / (2 × 표준편차)를 -1 ~ +1로 자른 연속 추세 신호."""
    mid, upper, _ = bollinger(close)
    return ((close - mid) / (upper - mid)).clip(-1, 1)


def _macd(close: pd.Series) -> pd.Series:
    line = ema(close, 12) - ema(close, 26)
    return np.sign(line - ema(line, 9))


SIGNALS = {
    "return_sign_20_60_120": lambda c: momentum.signal(c),  # 현재 전략
    "sma_cross_50_200": lambda c: np.sign(sma(c, 50) - sma(c, 200)),
    "price_vs_sma200": lambda c: np.sign(c - sma(c, 200)),
    "price_vs_sma_20_60_120": lambda c: pd.concat([np.sign(c - sma(c, n)) for n in (20, 60, 120)], axis=1).mean(
        axis=1, skipna=False
    ),
    "macd_12_26_9": _macd,
    "rsi14_above_50": lambda c: np.sign(rsi(c) - 50),
    "rsi14_reversal_30_70": lambda c: rsi_reversal_positions(rsi(c)),
    "donchian_55_20": donchian,
    "bb_breakout_20_2": lambda c: band_positions(c, breakout=True),
    "bb_reversion_20_2": lambda c: band_positions(c, breakout=False),
    "bb_percent_b_20": _percent_b,
}


# 결합 신호: 구성 신호의 동일 비중 평균 (롱/무포지션은 평균을 낸 뒤 음수를 0으로)
ENSEMBLES = {
    # 개별 비교에서 상위였던 추세 신호 — 결과를 보고 골라 선정 편향이 크다
    "ens_top_trend_6": [
        "return_sign_20_60_120", "price_vs_sma_20_60_120", "rsi14_above_50",
        "donchian_55_20", "bb_breakout_20_2", "bb_percent_b_20",
    ],
    # 방식이 다른 셋 (볼린저 비교 전에 제안한 조합)
    "ens_diverse_3": ["return_sign_20_60_120", "rsi14_above_50", "donchian_55_20"],
    # 역추세 둘만 빼고 전부 — 성적으로 고르지 않아 선정 편향이 작다
    "ens_all_trend_9": [n for n in SIGNALS if n not in ("rsi14_reversal_30_70", "bb_reversion_20_2")],
}


def signal(close: pd.Series, name: str) -> pd.Series:
    """개별 신호(SIGNALS) 또는 결합 신호(ENSEMBLES)."""
    if name in ENSEMBLES:
        return pd.concat([SIGNALS[m](close) for m in ENSEMBLES[name]], axis=1).mean(axis=1, skipna=False)
    return SIGNALS[name](close)


def position(close: pd.Series, name: str, long_only: bool) -> pd.Series:
    return size(close, signal(close, name), long_only)


def size(close: pd.Series, s: pd.Series, long_only: bool) -> pd.Series:
    """신호 × min(목표 변동성 / 최근 변동성, 최대 레버리지) — momentum.position과 같은 크기 규칙."""
    if long_only:
        s = s.clip(lower=0)
    vol = close.pct_change().rolling(momentum.VOL_WINDOW).std() * np.sqrt(365)
    return s * (momentum.TARGET_VOL / vol).clip(upper=momentum.MAX_LEVERAGE)
