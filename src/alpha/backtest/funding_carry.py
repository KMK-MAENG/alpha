"""xs-funding-carry: 횡단면 펀딩 캐리 롱숏 (역변동성 가중, 달러 중립, 주 1회 리밸런싱).

2026-10-06 에이전트 토론 워크플로우에서 사전 등록한 스펙 그대로 구현한 연구용 백테스트 (실거래 코드 아님).
설계·결과: docs/superpowers/specs/2026-10-06-funding-carry-research.md
개발 기간 판정만 한다. 보류 기간은 --holdout 을 줄 때만 한 번 계산·출력한다.
실행 (프로젝트 루트에서): uv run python -m alpha.backtest.funding_carry [--holdout]
"""
import json
import math
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import alpha.backtest.momentum as bt
from alpha.data import load_candles, load_funding, load_vision, vision_symbols
from alpha.strategies import carry, momentum
from alpha.strategies.carry import CAP, MIN_ELIG, MIN_LEG, delist_exit, k_of, rebalance_days, select  # noqa: F401
from alpha.strategies.momentum import UNIVERSE

EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"  # 최소 주문 점검용 (공개 API, 현재 규칙)
UTC = "UTC"
DEV_END = pd.Timestamp("2025-09-30", tz=UTC)
HO_START = pd.Timestamp("2025-10-01", tz=UTC)
CORR_START = pd.Timestamp("2020-06-14", tz=UTC)
COHORT_START = pd.Timestamp("2021-01-01", tz=UTC)
COHORT_CUTOFF = pd.Timestamp("2020-12-31", tz=UTC)

BASE = {"L": 7, "q": 0.20, "R": 7}
NEIGHBORS = [{"L": 4}, {"L": 11}, {"q": 0.10}, {"q": 0.30}, {"R": 4}, {"R": 10}]
ACCOUNT = 2000.0
DAY = pd.Timedelta(days=1)
N_BACKTESTS = 0


# ---------------------------------------------------------------- 데이터
def prep(candles: dict[str, pd.DataFrame], fundings: dict[str, pd.Series]) -> dict:
    closes = pd.DataFrame({s: c["close"] for s, c in candles.items()})
    volume = pd.DataFrame({s: c["volume"] for s, c in candles.items()})
    qv = pd.DataFrame({s: c["quote_volume"] for s, c in candles.items()})
    closes = closes.where(volume > 0)
    closes.index = closes.index.as_unit("ns")
    volume.index = qv.index = closes.index
    fl = {}
    for s, f in fundings.items():
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index).as_unit("ns").floor("min")
        fl[s] = f.sort_index()
    return {"closes": closes, "volume": volume, "qv": qv, "fundings": fl}


def load_live_universe() -> dict:
    return prep({s: load_candles("1d", s) for s in UNIVERSE}, {s: load_funding(s) for s in UNIVERSE})


def load_vision_all() -> dict:
    c, f = {}, {}
    for s in vision_symbols():
        c[s], f[s] = load_vision(s)
    return prep(c, f)


# ---------------------------------------------------------------- 신호·비중 (전략 본체는 alpha.strategies.carry)
def funding_signal(data: dict, L: int) -> pd.DataFrame:
    return carry.funding_signal(data["closes"], data["fundings"], L)


def eligibility(data: dict, S: pd.DataFrame, extra_mask: pd.DataFrame | None = None):
    return carry.eligibility(data["closes"], S, extra_mask)


def target_weights(data: dict, L: int, q: float, R: int, extra_mask=None) -> dict:
    return carry.carry_weights(data["closes"], data["fundings"], L, q, R, extra_mask)


def run_pnl(data: dict, W: pd.DataFrame, fee: float = 0.0005) -> pd.DataFrame:
    global N_BACKTESTS
    N_BACKTESTS += 1
    old = bt.FEE
    bt.FEE = fee
    try:
        pnl = bt.portfolio_pnl(data["closes"], data["fundings"], W)
    finally:
        bt.FEE = old
    pnl.index = pd.DatetimeIndex(pnl.index).as_unit("ns")
    return pnl.reindex(data["closes"].index, fill_value=0.0)


# ---------------------------------------------------------------- 지표
def metrics(total: pd.Series, W: pd.DataFrame | None, start, end) -> dict:
    r = total.loc[start:end]
    eq = (1 + r).cumprod()
    out = {"start": str(r.index[0].date()), "end": str(r.index[-1].date()),
           "sharpe": r.mean() / r.std() * np.sqrt(365), "ann_return": r.mean() * 365,
           "ann_vol": r.std() * np.sqrt(365), "mdd": (eq / eq.cummax() - 1).min()}
    if W is not None:
        w = W.loc[start:end]
        out["avg_gross_exposure"] = w.abs().sum(axis=1).mean()
        out["annual_turnover"] = W.diff().abs().sum(axis=1).loc[start:end].mean() * 365
    return out


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(365)


def daily_funding(data: dict) -> pd.DataFrame:
    idx = data["closes"].index
    out = {}
    for s in data["closes"].columns:
        f = data["fundings"].get(s)
        if f is None or len(f) == 0:
            out[s] = pd.Series(0.0, index=idx)
        else:
            out[s] = f.resample("1D").sum().reindex(idx, fill_value=0.0)
    return pd.DataFrame(out, index=idx)


def drift_pnl(data: dict, W: pd.DataFrame, days: pd.DatetimeIndex, fee: float = 0.0005) -> pd.Series:
    """수량 고정: 리밸런싱일 목표 비중으로 명목을 고정, 이후 가격·자산 변화로 비중이 흐른다.
    리밸런싱일(또는 상장 폐지 청산) 종가에 거래, 수수료 = fee × Σ|w_새 − w_흘러간|, 그날 수익에서 뺀다."""
    global N_BACKTESTS
    N_BACKTESTS += 1
    rets = data["closes"].pct_change(fill_method=None).fillna(0.0).to_numpy()
    fund = daily_funding(data).to_numpy()
    Wn = W.to_numpy()
    is_reb = W.index.isin(days)
    held = np.zeros(W.shape[1])
    out = np.zeros(len(W))
    for i in range(len(W)):
        gross = float(held @ (rets[i] - fund[i]))
        drifted = held * (1 + rets[i]) / (1 + gross)
        if is_reb[i]:
            new = Wn[i].copy()
        else:
            new = drifted.copy()
            exit_ = (Wn[i] == 0) & (drifted != 0)  # 상장 폐지 청산
            new[exit_] = 0.0
        f = fee * np.abs(new - drifted).sum()
        out[i] = gross - f
        held = new
    return pd.Series(out, index=W.index)


def newey_west_ols(y: np.ndarray, X: np.ndarray):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    lag = int(math.floor(4 * (n / 100) ** (2 / 9)))
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, beta / se, lag


def reversal_factor(data: dict, tw: dict, q: float) -> pd.Series:
    """1주 반전: 같은 적격 집합·같은 리밸런싱일, R7 하위 k 롱 +0.5 동일 비중, 상위 k 숏 −0.5. 가격 수익만."""
    closes = data["closes"]
    r7 = closes / closes.shift(7) - 1
    rows = {}
    for t in tw["days"]:
        w = pd.Series(0.0, index=closes.columns)
        syms = tw["elig"].columns[tw["elig"].loc[t].to_numpy()]
        if len(syms) >= MIN_ELIG:
            s = r7.loc[t, syms].dropna()
            k = k_of(q, len(s))
            top, bottom = select(s, k)
            if len(top.intersection(bottom)) == 0:
                w[bottom] = 0.5 / len(bottom)
                w[top] = -0.5 / len(top)
        rows[t] = w
    W = pd.DataFrame(rows).T.reindex(closes.index).ffill().fillna(0.0)
    W, _ = delist_exit(W, closes, tw["days"])
    return run_pnl(data, W, fee=0.0)["price"]


def deflated_sharpe(r: pd.Series, trial_sr_ann: list[float], n_trials: int) -> float:
    sr = r.mean() / r.std()
    v = np.var(np.array(trial_sr_ann) / np.sqrt(365), ddof=1)
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf(z))


def min_order_shortfall(data: dict, tw: dict, start, end) -> tuple[float, pd.Series]:
    info = json.load(urllib.request.urlopen(EXCHANGE_INFO, timeout=30))
    rules = {}
    for s in info["symbols"]:
        f = {x["filterType"]: x for x in s["filters"]}
        rules[s["symbol"]] = (float(f["MIN_NOTIONAL"]["notional"]), float(f["LOT_SIZE"]["minQty"]))
    rows = {}
    for t in tw["days"]:
        if not (start <= t <= end):
            continue
        w = tw["W"].loc[t]
        w = w[w != 0]
        tot = 0.0
        for s, x in w.items():
            mn, mq = rules.get(s, (np.inf, 0))
            need = max(mn, mq * data["closes"].at[t, s])
            if abs(x) * ACCOUNT < need:
                tot += abs(x)
        rows[t] = tot
    ser = pd.Series(rows)
    yearly = ser.groupby(ser.index.year).mean()
    missing = sorted({s for s in UNIVERSE if s not in rules})
    return float(yearly.mean()), yearly, missing


# ---------------------------------------------------------------- 메인
def main(run_holdout: bool) -> None:
    res = {}
    data = load_live_universe()
    print(f"[데이터] 56개, {data['closes'].index[0].date()} ~ {data['closes'].index[-1].date()}")

    # 1) 기본 개발 기간
    base = target_weights(data, **BASE)
    pnl = run_pnl(data, base["W"])
    start = base["start"]
    dev = metrics(pnl["total"], base["W"], start, DEV_END)
    res["dev"] = dev
    print(f"\n[1 개발 기간 기본] 시작(적격 15 이상 첫날) {start.date()}")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in dev.items()})
    d = pnl.loc[start:DEV_END]
    cost = {c: d[c].mean() * 365 for c in ["price", "funding", "fee"]}
    print("손익 분해(연평균, 일 수익 평균×365):", {k: round(v, 4) for k, v in cost.items()})
    res["cost"] = cost

    # 3) 이웃
    trial_srs = [dev["sharpe"]]
    neigh = []
    for nb in NEIGHBORS:
        p = BASE | nb
        tw = target_weights(data, **p)
        pn = run_pnl(data, tw["W"])
        sr = metrics(pn["total"], tw["W"], tw["start"], DEV_END)["sharpe"]
        trial_srs.append(sr)
        neigh.append((f"L={p['L']},q={p['q']:.0%},R={p['R']}", sr))
        print(f"[3 이웃] {neigh[-1][0]} 시작 {tw['start'].date()} Sharpe {sr:.3f}")
    res["neighbors"] = neigh

    # 4) 수수료 2배
    pnl2 = run_pnl(data, base["W"], fee=0.001)
    sr_fee2 = sharpe(pnl2["total"].loc[start:DEV_END])
    print(f"\n[4 수수료 2배] 개발 Sharpe {sr_fee2:.3f}")
    res["fee2"] = sr_fee2

    # 미래 참조 점검: 비중 하루 더 늦춤
    pnl_lag = run_pnl(data, base["W"].shift(1).fillna(0.0))
    sr_lag = sharpe(pnl_lag["total"].loc[start:DEV_END])
    print(f"[미래 참조 점검] 비중 하루 지연 개발 Sharpe {sr_lag:.3f}")
    res["lag"] = sr_lag

    # 5) 생존 편향: vision 코호트
    vdata = load_vision_all()
    first = vdata["closes"].apply(pd.Series.first_valid_index)
    cohort = sorted(s for s, f in first.items() if pd.notna(f) and f <= COHORT_CUTOFF)
    vlast = vdata["closes"].index[-1]
    delisted = [s for s in cohort if vdata["closes"][s].last_valid_index() < vlast - pd.Timedelta(days=7)]
    cdata = {"closes": vdata["closes"][cohort], "volume": vdata["volume"][cohort], "qv": vdata["qv"][cohort],
             "fundings": {s: vdata["fundings"][s] for s in cohort}}
    no_fund = [s for s in cohort if len(cdata["fundings"][s]) == 0]
    ctw = target_weights(cdata, **BASE)
    cpnl = run_pnl(cdata, ctw["W"])
    sr_cohort = sharpe(cpnl["total"].loc[COHORT_START:DEV_END])
    cm = metrics(cpnl["total"], ctw["W"], COHORT_START, DEV_END)
    print(f"\n[5 코호트] vision 첫 유효 종가 ≤ 2020-12-31: {len(cohort)}개 (이후 폐지 {len(delisted)}개, "
          f"펀딩 없음 {len(no_fund)}), 2021-01-01~2025-09-30 Sharpe {sr_cohort:.3f}, 연 {cm['ann_return']:.4f}, "
          f"MDD {cm['mdd']:.4f}, 상장 폐지/정지 청산 {ctw['n_exit']}건 (코호트 전 기간)")
    n_exit_dev = None
    res["cohort"] = (len(cohort), len(delisted), sr_cohort, ctw["n_exit"])
    # 참고: 시점별 30일 평균 quote_volume 상위 60
    qv30 = vdata["qv"].where(vdata["volume"] > 0).rolling(30, min_periods=1).mean()
    top60 = qv30.rank(axis=1, ascending=False, method="min") <= 60
    ttw = target_weights(vdata, **BASE, extra_mask=top60)
    tpnl = run_pnl(vdata, ttw["W"])
    sr_top60 = sharpe(tpnl["total"].loc[COHORT_START:DEV_END])
    sr_top60_full = sharpe(tpnl["total"].loc[ttw["start"]:DEV_END])
    print(f"[5 참고] 거래대금 상위 60 (679 심볼) 2021-01-01~2025-09-30 Sharpe {sr_top60:.3f}, "
          f"{ttw['start'].date()}~ Sharpe {sr_top60_full:.3f}, 청산 {ttw['n_exit']}건")
    res["top60"] = (sr_top60, sr_top60_full)

    # 6) 상관
    live = run_pnl(data, momentum.combined_weights(data["closes"], **momentum.LIVE_OPTIONS))["total"]  # 실거래 모멘텀
    live.index = pd.DatetimeIndex(live.index).as_unit("ns")
    live = live.loc[CORR_START:DEV_END]
    y = pnl["total"].reindex(live.index).fillna(0.0)
    corr = y.corr(live)
    print(f"\n[6 상관] 실거래 모멘텀과 {live.index[0].date()}~{live.index[-1].date()} ({len(live)}일) 상관 {corr:.3f}")
    res["corr"] = corr

    # 8) 수량 고정
    dr = drift_pnl(data, base["W"], base["days"])
    sr_drift = sharpe(dr.loc[start:DEV_END])
    print(f"[8 수량 고정] 개발 Sharpe {sr_drift:.3f} (기본 {dev['sharpe']:.3f})")
    res["drift"] = sr_drift

    # 9) 회귀 절편
    rets = data["closes"].pct_change(fill_method=None)
    mkt = rets.mean(axis=1, skipna=True).fillna(0.0)
    rev = reversal_factor(data, base, BASE["q"])
    X = pd.DataFrame({"live": live, "mkt": mkt.reindex(live.index), "rev1w": rev.reindex(live.index)}).fillna(0.0)
    beta, tval, lag = newey_west_ols(y.to_numpy(), X.to_numpy())
    print(f"[9 회귀] 일 절편 {beta[0]:.6f} (연 {beta[0] * 365:.4f}), NW t {tval[0]:.2f} (lag {lag}); "
          f"베타 live {beta[1]:.3f} (t {tval[1]:.2f}), mkt {beta[2]:.3f} (t {tval[2]:.2f}), "
          f"rev1w {beta[3]:.3f} (t {tval[3]:.2f})")
    print(f"   반전 팩터 자체 Sharpe {sharpe(rev.loc[CORR_START:DEV_END]):.3f}, 캐리와 상관 "
          f"{y.corr(X['rev1w']):.3f}")
    res["reg"] = (beta, tval)

    # 특유 점검
    print("\n[특유 점검]")
    r = pnl["total"].loc[start:DEV_END]
    yearly_sr = r.groupby(r.index.year).apply(sharpe)
    yearly_ret = (1 + r).groupby(r.index.year).prod() - 1
    print("연도별 Sharpe:", yearly_sr.round(3).to_dict())
    print("연도별 수익(복리):", yearly_ret.round(4).to_dict())
    res["yearly"] = (yearly_sr, yearly_ret)
    info = base["info"]
    act = info[(info.index >= start) & (info.index <= DEV_END) & (info["N"] >= MIN_ELIG)]
    med = act["xs_std"].median()
    seg = pd.Series(base["W"].index.isin(base["days"]), index=base["W"].index).cumsum()
    seg_start = pd.Series(base["days"], index=range(1, len(base["days"]) + 1))
    day_reb = seg.map(seg_start)
    hi_reg = day_reb.map(info["xs_std"]) > med
    lo_reg = day_reb.map(info["xs_std"]) <= med
    rr = pnl["total"]
    sl = slice(start, DEV_END)
    print(f"펀딩 횡단면 표준편차 중앙값 {med:.6f}: 위 국면 Sharpe {sharpe(rr[hi_reg].loc[sl]):.3f} "
          f"({hi_reg.loc[sl].sum()}일), 아래 국면 Sharpe {sharpe(rr[lo_reg].loc[sl]):.3f} ({lo_reg.loc[sl].sum()}일)")
    weekly = (1 + rr.loc[sl]).groupby(seg.loc[sl]).prod() - 1
    print(f"최악의 주(리밸런싱 구간) {weekly.min():.4f} (구간 시작 {seg_start.get(weekly.idxmin())})")
    # 숏 다리 단일 종목 최악 주간 손실 (계좌 기준)
    worst = (0.0, None, None)
    for s in base["W"].columns[(base["W"] < 0).any()]:
        p = bt.backtest(data["closes"][s].loc[data["closes"][s].first_valid_index():],
                        data["fundings"][s], base["W"][s].loc[data["closes"][s].first_valid_index():])["total"]
        p = p.reindex(rr.index, fill_value=0.0).loc[sl]
        held_short = base["W"][s].shift(1).loc[sl] < 0
        wk = p.where(held_short, 0.0).groupby(seg.loc[sl]).sum()
        if wk.min() < worst[0]:
            worst = (wk.min(), s, seg_start.get(wk.idxmin()))
    print(f"숏 다리 단일 종목 최악 주간 손실(계좌 기준) {worst[0]:.4f} ({worst[1]}, 구간 시작 {worst[2]})")
    print(f"왜도 {skew(r):.3f}, 첨도(초과) {kurtosis(r):.3f}, MDD {dev['mdd']:.4f}")
    print(f"리밸런싱 {len(act)}회: 경계 동률 비율 {act['tie'].mean():.3f}, 다리 겹침 무포지션 비율 "
          f"{act['overlap'].mean():.3f}, 평균 롱 {act.loc[act['active'], 'n_long'].mean():.2f}개 / 숏 "
          f"{act.loc[act['active'], 'n_short'].mean():.2f}개, 평균 적격 {act['N'].mean():.1f}")
    short_avg, short_y, missing = min_order_shortfall(data, base, start, DEV_END)
    print(f"2,000 USDT 최소 주문 미달 비중 합 (리밸런싱 평균의 연평균) {short_avg:.4f}; 연도별 "
          f"{short_y.round(4).to_dict()}; 거래 규칙 없는 심볼 {missing}")
    print(f"연 회전율 {dev['annual_turnover']:.2f}, 평균 총 노출 {dev['avg_gross_exposure']:.3f}, "
          f"최대 총 노출 {base['W'].abs().sum(axis=1).loc[sl].max():.3f}")
    print(f"56개 상장 폐지/정지 청산 {base['n_exit']}건 (전 기간), vision 코호트 {ctw['n_exit']}건")
    # Deflated Sharpe, t
    years = len(r) / 365
    print(f"t값 Sharpe×√년수 = {dev['sharpe']:.3f}×√{years:.2f} = {dev['sharpe'] * np.sqrt(years):.2f}")
    for n in (7, 21):
        print(f"Deflated Sharpe (시도 {n}, 분산=이 전략 7개 시도 Sharpe) {deflated_sharpe(r, trial_srs, n):.3f}")
    res["dsr"] = {n: deflated_sharpe(r, trial_srs, n) for n in (7, 21)}

    # 판정
    nb_ok = all(s > 0.3 for _, s in neigh)
    checks = {
        "1 개발 Sharpe>=0.7": dev["sharpe"] >= 0.7,
        "3 이웃 모두>0.3": nb_ok,
        "4 수수료2배>=0.5": sr_fee2 >= 0.5,
        "5 코호트>0.3": sr_cohort > 0.3,
        "6 상관<0.5": corr < 0.5,
        "8 수량고정>=0.5": sr_drift >= 0.5,
        "9 절편>0": beta[0] > 0,
        "7 총노출<=1.5": base["W"].abs().sum(axis=1).max() <= 1.5,
    }
    print("\n[개발 기간 판정]", checks)
    all_dev = all(checks.values())
    print("개발 기간 전부 통과:", all_dev)

    if run_holdout:
        if not all_dev:
            print("개발 기간 불합격 → 보류 기간을 돌리지 않는다")
        else:
            ho = metrics(pnl["total"], base["W"], HO_START, pnl.index[-1])
            print("\n[보류 기간]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ho.items()})
    print(f"\n총 백테스트(손익 계산) 횟수 {N_BACKTESTS}")


if __name__ == "__main__":
    main("--holdout" in sys.argv)
