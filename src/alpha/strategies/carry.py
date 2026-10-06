"""횡단면 펀딩 캐리 롱숏 + 숏 스퀴즈 방어 (실거래·백테스트 공용).

- 신호: 리밸런싱일(일요일 봉) 마감 시각 T(= 월요일 00:00 UTC) 기준 [T − 7일, T) 펀딩 합. T 정산은 넣지 않는다.
  창 안 기록이 12시간 넘게 끊기면 NaN
- 적격: 첫 종가 후 60일 이상, 그날 종가·신호·60일 변동성 있음. 적격 15개 미만이면 무포지션
- 선정: 펀딩 상위 20%(최소 3, 경계 동률 포함) 숏, 하위 20% 롱. 다리 안 60일 변동성 역수 가중, 롱 +0.5 / 숏 −0.5,
  종목당 |비중| ≤ 0.15 (넘는 몫은 현금). 주 1회 리밸런싱, 사이에는 비중 유지
- 숏 스퀴즈 방어(LIVE_OPTIONS): 종가가 60일 고점의 1σ(60일 일 변동성) 안인 코인은 숏 후보에서 빼고 다음 순위로 채운다.
  주중에 보유 숏이 그 상태가 되면 다음 리밸런싱까지 청산(현금, 대체 없음)
- 보유 중 다음 날 종가가 없으면(상장 폐지) 그날 청산 — 다음 날을 아는 백테스트용 근사 (마지막 날에는 적용 안 됨)
연구·검증: docs/superpowers/specs/2026-10-06-funding-carry-research.md
"""
import math

import numpy as np
import pandas as pd

from alpha.strategies.momentum import VOL_WINDOW

LOOKBACK = 7  # 펀딩 합 창 (일)
QUANTILE = 0.20
REBALANCE_DAYS = 7
MIN_LEG, MIN_ELIG, CAP, AGE_DAYS = 3, 15, 0.15, 60
GUARD_WINDOW, GUARD_MIN_PERIODS, GUARD_SIGMAS = 60, 40, 1.0
LIVE_OPTIONS = {"squeeze_guard": True}
DAY = pd.Timedelta(days=1)
H12 = pd.Timedelta(hours=12)


def funding_signal(closes: pd.DataFrame, fundings: dict[str, pd.Series], lookback: int = LOOKBACK) -> pd.DataFrame:
    """S[t, i] = [T − lookback일, T) 안 펀딩 합 (10자리 반올림), T = t + 1일. 창 시작·끝 12시간 안에 기록이 없거나
    창 안 기록 간격이 12시간을 넘으면 NaN."""
    idx = closes.index.as_unit("ns")
    T = (idx + DAY).asi8
    lo_t, ok_first, ok_last = T - lookback * DAY.value, T - lookback * DAY.value + H12.value, T - H12.value
    out = {}
    for s in closes.columns:
        res = np.full(len(T), np.nan)
        f = fundings.get(s)
        if f is None or len(f) == 0:
            out[s] = res
            continue
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index).as_unit("ns").floor("min")  # 정산 시각의 ms 오차 제거
        f = f.sort_index()
        ft, v = f.index.asi8, f.to_numpy(dtype=float)
        lo = np.searchsorted(ft, lo_t, "left")
        hi = np.searchsorted(ft, T, "left")
        bad = np.concatenate([[0], np.cumsum(np.diff(ft) > H12.value)])  # bad[j] = j번째 이전 간격 중 12시간 초과 수
        has = hi > lo
        lo_c, hi_c = np.clip(lo, 0, len(ft) - 1), np.clip(hi - 1, 0, len(ft) - 1)
        ok = has & (ft[lo_c] <= ok_first) & (ft[hi_c] >= ok_last) & ((bad[hi_c] - bad[lo_c]) == 0)
        for j in np.flatnonzero(ok):
            res[j] = round(float(np.sum(v[lo[j] : hi[j]])), 10)
        out[s] = res
    return pd.DataFrame(out, index=closes.index)


def eligibility(closes: pd.DataFrame, S: pd.DataFrame, extra_mask: pd.DataFrame | None = None):
    sigma = closes.pct_change(fill_method=None).rolling(VOL_WINDOW, min_periods=40).std()
    first = closes.apply(pd.Series.first_valid_index)
    age_ok = pd.DataFrame({s: (closes.index - first[s]) >= pd.Timedelta(days=AGE_DAYS) if pd.notna(first[s])
                           else np.zeros(len(closes), bool) for s in closes.columns}, index=closes.index)
    elig = age_ok & closes.notna() & S.notna() & sigma.notna() & (sigma > 0)
    if extra_mask is not None:
        elig &= extra_mask.reindex_like(elig).fillna(False).astype(bool)
    return elig, sigma


def rebalance_days(elig: pd.DataFrame, R: int = REBALANCE_DAYS) -> pd.DatetimeIndex:
    idx = elig.index
    if R == 7:
        return idx[idx.dayofweek == 6]
    cnt = elig.sum(axis=1)
    first = cnt[cnt >= MIN_ELIG].index[0]
    anchor = idx[(idx >= first) & (idx.dayofweek == 6)][0]
    return idx[(idx >= anchor) & (((idx - anchor).days % R) == 0)]


def k_of(q: float, n: int) -> int:
    return max(MIN_LEG, math.ceil(round(q * n, 9)))


def select(s: pd.Series, k: int):
    """경계 동률 포함: 숏 = S >= k번째 큰 값, 롱 = S <= k번째 작은 값."""
    vals = np.sort(s.to_numpy())
    hi_thr, lo_thr = vals[-k], vals[k - 1]
    return s.index[s >= hi_thr], s.index[s <= lo_thr]


def delist_exit(W: pd.DataFrame, closes: pd.DataFrame, days: pd.DatetimeIndex):
    """보유 중 다음 날 종가가 없으면 그날(마지막 유효 종가일)부터 구간 끝까지 비중 0."""
    seg = pd.Series(W.index.isin(days), index=W.index).cumsum()
    nxt = closes.shift(-1).isna()
    nxt.iloc[-1] = False
    flag = (W != 0) & nxt
    cum = flag.groupby(seg).cummax()
    n_exit = int(flag.groupby(seg).any().to_numpy().sum())
    return W.mask(cum, 0.0), n_exit


def squeeze_state(closes: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """종가가 60일 고점(당일 포함)의 1σ 안이면 True."""
    high = closes.rolling(GUARD_WINDOW, min_periods=GUARD_MIN_PERIODS).max()
    ok = closes.notna() & high.notna() & sigma.notna()
    return ((closes >= high * (1 - GUARD_SIGMAS * sigma)) & ok).astype(bool)


def carry_weights(closes: pd.DataFrame, fundings: dict[str, pd.Series], lookback: int = LOOKBACK, q: float = QUANTILE,
                  R: int = REBALANCE_DAYS, extra_mask: pd.DataFrame | None = None, squeeze_guard: bool = False) -> dict:
    """날짜 × 코인 목표 비중 W와 진단 정보. 날짜 t 행은 t 종가 기준 (t+1부터 보유)."""
    S = funding_signal(closes, fundings, lookback)
    elig, sigma = eligibility(closes, S, extra_mask)
    days = rebalance_days(elig, R)
    squeezed = squeeze_state(closes, sigma) if squeeze_guard else None
    rows, info = {}, []
    for t in days:
        syms = elig.columns[elig.loc[t].to_numpy()]
        n = len(syms)
        w = pd.Series(0.0, index=closes.columns)
        rec = {"t": t, "N": n, "active": False, "overlap": False, "tie": False, "guard_flat": False, "n_long": 0,
               "n_short": 0, "n_excl": 0, "xs_std": np.nan}
        if n >= MIN_ELIG:
            s = S.loc[t, syms]
            rec["xs_std"] = s.std()
            k = k_of(q, n)
            short, long_ = select(s, k)
            rec["tie"] = len(short) > k or len(long_) > k
            if squeeze_guard:
                rec["n_excl"] = int(squeezed.loc[t, short].sum())
                cand = syms[~squeezed.loc[t, syms].to_numpy()]
                if len(cand) < k:
                    rec["guard_flat"] = True
                    rows[t] = w
                    info.append(rec)
                    continue
                short = select(S.loc[t, cand], k)[0]
            if len(short.intersection(long_)) > 0:
                rec["overlap"] = True
            else:
                inv = 1.0 / sigma.loc[t, syms]
                w[long_] = 0.5 * inv[long_] / inv[long_].sum()
                w[short] = -0.5 * inv[short] / inv[short].sum()
                w = w.clip(-CAP, CAP)
                rec |= {"active": True, "n_long": len(long_), "n_short": len(short)}
        rows[t] = w
        info.append(rec)
    W = pd.DataFrame(rows).T.reindex(closes.index).ffill().fillna(0.0)
    if squeeze_guard:  # 주중 감시: 보유 숏이 스퀴즈 상태가 되면 그 구간 끝까지 0
        isreb = W.index.isin(days)
        seg = pd.Series(isreb, index=W.index).cumsum()
        flag = (W < 0) & squeezed
        flag.loc[isreb] = False
        W = W.mask(flag.groupby(seg).cummax(), 0.0)
    W, n_exit = delist_exit(W, closes, days)
    cnt = elig.sum(axis=1)
    start = cnt[cnt >= MIN_ELIG].index[0] if (cnt >= MIN_ELIG).any() else None
    info = pd.DataFrame(info).set_index("t") if info else pd.DataFrame()
    return {"W": W, "days": days, "info": info, "n_exit": n_exit, "start": start, "S": S, "elig": elig, "sigma": sigma}
