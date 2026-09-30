"""QQQ 일봉으로 '오늘 밤 오버나잇을 들지 말지' 진입 신호를 발굴하고, QQQUSDT 선물 오버나잇에 적용한다.

- 목표: 다음 밤 수익률 = 다음 정규장 시가 / 오늘 종가 - 1. 후보는 오늘 종가 시점까지의 정보만 쓴다
  (VIX 종가는 16:15 ET로 QQQ 종가보다 늦어 전날 값만 쓴다)
- 1) train에서 후보별 예측력(t값) → 2) valid에서 같은 방향으로 재현된 후보만 채택
  → 3) 선형회귀로 밤 수익률을 예측해 왕복 수수료(COST)를 넘는 밤만 보유 (조정할 임계값 없음)
  → 4) train+valid로 다시 맞춘 모델로 test를 한 번 평가
- --gp: 3) 대신 GP로 보유 규칙(수식 > 0이면 보유)을 탐색한다. 연속 후보는 252일 z-score로 표준화하고
  이산 후보(요일·월말 등)는 그대로 재료로 쓴다. 함수는 원소별·조건 중심 + 일봉 윈도우(5·20·60일) 시계열 소수만 쓰고,
  train 밤 단위 수수료 차감 Sharpe로 진화 → valid Sharpe·상관 필터로 최대 10개 → 보유 비율 평균을 포지션으로 결합.
  몇 밤만 운 좋게 맞히는 규칙을 막기 위해 train·valid 모두 밤의 MIN_HOLD 이상 보유해야 한다
- 마지막으로 같은 신호로 QQQUSDT 선물 오버나잇을 1분봉 백테스트한다

실행 (프로젝트 루트에서): uv run python -m alpha.mining.overnight_timing [--gp] [--pop-size N] [--generations N] [--seed N]
"""
import argparse

import exchange_calendars as xc
import numpy as np
import pandas as pd

from alpha.backtest import overnight
from alpha.data import load_candles, load_funding, load_yahoo_daily
from alpha.mining import expr, gplearn_search, search, wq101

COST = 0.001  # 왕복 테이커 수수료 (0.05% x 2)
SPLITS = {"train": ("1999-01-01", "2012-12-31"), "valid": ("2013-01-01", "2019-12-31"), "test": ("2020-01-01", None)}
MIN_T_TRAIN, MIN_T_VALID = 2.0, 1.0
Z_WINDOW = 252  # GP 재료 표준화 윈도우 (1년)
MIN_HOLD = 0.05  # GP 규칙의 최소 보유 밤 비율 (월 1회꼴)
DISCRETE = ["days_to_next_session", "month_end", "opex", "mon", "tue", "wed", "thu", "fri"]  # 표준화하지 않는다
GP_WINDOWS = [5, 20, 60]  # 거래일 기준 1주·1달·1분기
GP_FUNCTIONS = (
    ["add", "sub", "mul", "div", "neg", "abs", "max", "min"]
    + [f for f in wq101.FUNCTIONS if getattr(f, "name", None) in ("sign", "lt", "where")]
    + [
        gplearn_search.ts_function(name, expr.OPS[name][0], 1, d)
        for name in ("ts_mean", "ts_zscore", "ts_rank", "ts_max", "ts_min")
        for d in GP_WINDOWS
    ]
)


def features(qqq: pd.DataFrame, vix: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """날짜 t의 행은 t 종가 시점까지 알 수 있는 값만 담는다."""
    c, o, h, l, v = (qqq[k] for k in ["close", "open", "high", "low", "volume"])
    r = c.pct_change()
    on = o / c.shift(1) - 1
    vix_close = vix["close"].reindex(c.index).ffill().shift(1)
    sign = np.sign(r)
    next_session = sessions[sessions.searchsorted(c.index, side="right")]
    dow = c.index.dayofweek
    X = pd.DataFrame(
        {
            "intraday": c / o - 1,
            "overnight": on,
            "ret_1d": r,
            "ret_5d": c.pct_change(5),
            "ret_20d": c.pct_change(20),
            "ma50_gap": c / c.rolling(50).mean() - 1,
            "ma200_gap": c / c.rolling(200).mean() - 1,
            "drawdown_52w": c / c.rolling(252).max() - 1,
            "close_location": (c - l) / (h - l),
            "range": (h - l) / c,
            "streak": sign.groupby((sign != sign.shift()).cumsum()).cumsum(),
            "overnight_mean_20d": on.rolling(20).mean(),
            "overnight_mean_60d": on.rolling(60).mean(),
            "vol_20d": r.rolling(20).std(),
            "vol_ratio": r.rolling(5).std() / r.rolling(60).std(),
            "vix": vix_close,
            "vix_change": vix_close.pct_change(),
            "vix_vs_ma20": vix_close / vix_close.rolling(20).mean() - 1,
            "volume_surge": v / v.rolling(20).mean(),
            "days_to_next_session": (next_session - c.index).days,
            "month_end": c.index.month != next_session.month,
            "opex": (dow == 4) & (c.index.day >= 15) & (c.index.day <= 21),
        }
        | {day: dow == k for k, day in enumerate(["mon", "tue", "wed", "thu", "fri"])},
        index=c.index,
    )
    return X.astype("float64").replace([np.inf, -np.inf], np.nan)


def target(qqq: pd.DataFrame) -> pd.Series:
    return qqq["open"].shift(-1) / qqq["close"] - 1


def _period(x, name):
    start, end = SPLITS[name]
    return x.loc[start:end]


def _t(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    r = d.iloc[:, 0].corr(d.iloc[:, 1])
    return r * np.sqrt((len(d) - 2) / (1 - r**2))


def screen(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """후보별 다음 밤 수익률과의 상관 t값 (train, valid)."""
    return pd.DataFrame(
        {name: [_t(_period(X[c], name), _period(y, name)) for c in X] for name in ["train", "valid"]},
        index=X.columns,
    )


def select(table: pd.DataFrame) -> list[str]:
    ok = (
        (table["train"].abs() >= MIN_T_TRAIN)
        & (np.sign(table["train"]) == np.sign(table["valid"]))
        & (table["valid"].abs() >= MIN_T_VALID)
    )
    return list(table.index[ok])


def fit_predict(X: pd.DataFrame, y: pd.Series, columns: list[str], fit_periods: list[str]) -> pd.Series:
    """fit_periods 데이터로 표준화·선형회귀를 맞추고 전체 기간의 다음 밤 수익률을 예측한다."""
    data = pd.concat([_period(pd.concat([X[columns], y.rename("y")], axis=1), p) for p in fit_periods]).dropna()
    mean, std = data[columns].mean(), data[columns].std()
    design = np.column_stack([np.ones(len(data)), ((data[columns] - mean) / std).to_numpy()])
    coef, *_ = np.linalg.lstsq(design, data["y"].to_numpy(), rcond=None)
    return coef[0] + ((X[columns] - mean) / std) @ coef[1:]


def nightly_pnl(hold: pd.Series, y: pd.Series) -> pd.Series:
    return hold.astype(float) * (y - COST)


def _stats(pnl: pd.Series, hold: pd.Series) -> dict:
    held = pnl[hold > 0]
    return {
        "연수익": pnl.mean() * 252,
        "Sharpe": pnl.mean() / pnl.std() * np.sqrt(252),
        "보유 밤 비율": (hold > 0).mean(),
        "보유 밤 평균(순)": held.mean(),
        "보유 밤 적중률": (held > 0).mean(),
    }


def _sharpe(pnl: pd.Series) -> float:
    std = pnl.std()
    return pnl.mean() / std * np.sqrt(252) if std > 0 else 0.0


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """연속 피처는 과거 Z_WINDOW일 z-score로, 이산 피처(DISCRETE)는 그대로 둔다."""
    continuous = [c for c in X if c not in DISCRETE]
    z = X.copy()
    z[continuous] = (X[continuous] - X[continuous].rolling(Z_WINDOW).mean()) / X[continuous].rolling(Z_WINDOW).std()
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _gp_fitness(y, y_pred, sample_weight):
    hold = y_pred > 0
    return _sharpe(pd.Series(hold * (y - COST))) if hold.mean() >= MIN_HOLD else 0.0


def gp_search(X: pd.DataFrame, y: pd.Series, pop_size: int, generations: int, seed: int):
    """(선택된 규칙별 구간 Sharpe 표 + 결합 "combo" 행, 결합 포지션 0~1)."""
    Z = standardize(X)
    programs = gplearn_search.evolve(
        _period(Z, "train"), _period(y, "train").to_numpy(), _gp_fitness,
        pop_size, generations, seed, GP_FUNCTIONS, const_range=(-2.0, 2.0),
    )
    holds = {str(p): pd.Series(p.execute(Z.to_numpy()) > 0, index=Z.index) for p in programs[: search.N_CANDIDATES]}

    scored = [
        (_sharpe(_period(nightly_pnl(h, y), "valid")), name)
        for name, h in holds.items()
        if _period(h, "valid").mean() >= MIN_HOLD
    ]
    passed = sorted((s for s in scored if s[0] >= search.MIN_VALID_SHARPE), reverse=True)
    until_valid = lambda name: holds[name].loc[: SPLITS["valid"][1]].astype(float)
    chosen = []
    for _, name in passed:
        if all(abs(until_valid(name).corr(until_valid(c))) < search.MAX_CORR for c in chosen):
            chosen.append(name)
        if len(chosen) == search.MAX_ALPHAS:
            break

    combo = pd.concat([holds[c] for c in chosen], axis=1).mean(axis=1) if chosen else pd.Series(0.0, index=y.index)
    rows = {name: holds[name] for name in chosen} | {"combo": combo}
    table = pd.DataFrame.from_dict(
        {name: {p: _sharpe(_period(nightly_pnl(h, y), p)) for p in SPLITS} | {"평균 포지션": h.mean()} for name, h in rows.items()},
        orient="index",
    )
    return table, combo


def _hypothesis_hold(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    table = screen(X, y)
    chosen = select(table)
    print(f"\n[후보별 다음 밤 수익률 예측력 (t값)]\n{table.round(2).sort_values('train').to_string()}")
    print(f"\n채택 (train |t|≥{MIN_T_TRAIN}, valid 같은 방향 |t|≥{MIN_T_VALID}): {chosen}")
    pred = pd.concat(
        [
            fit_predict(X, y, chosen, ["train"]).loc[: SPLITS["valid"][1]],
            fit_predict(X, y, chosen, ["train", "valid"]).loc[SPLITS["test"][0] :],
        ]
    )
    return pred > COST


def _etf_report(hold: pd.Series, y: pd.Series, qqq: pd.DataFrame) -> pd.DataFrame:
    strategies = {"signal": hold, "every_night": pd.Series(True, index=y.index)}
    rows = {}
    for name in SPLITS:
        for strat, h in strategies.items():
            rows[(name, strat)] = _stats(_period(nightly_pnl(h, y), name), _period(h, name))
        close_to_close = _period(qqq["close"].pct_change().reindex(y.index), name)
        rows[(name, "buy_and_hold")] = {"연수익": close_to_close.mean() * 252, "Sharpe": _sharpe(close_to_close)}
    return pd.DataFrame.from_dict(rows, orient="index")


def _perp_report(hold: pd.Series) -> pd.DataFrame:
    """날짜 t 종가의 판단을 QQQUSDT 선물의 t 마감 이후 장외 구간 전체에 쓴다."""
    candles = load_candles("1m", "QQQUSDT")
    funding = load_funding("QQQUSDT").loc[candles.index[0] :]
    schedule = overnight.nyse_schedule(candles.index[0], candles.index[-1])
    outside = overnight.label_sessions(candles.index, schedule) != "rth"
    held = hold.astype(float).reindex(overnight.session_dates(candles.index, schedule)).fillna(0.0).to_numpy()
    positions = {"signal": outside * held, "every_night": outside, "buy_and_hold": np.ones(len(candles), bool)}
    return pd.DataFrame.from_dict(
        {
            name: overnight.summary(
                overnight.backtest(candles, funding, pd.Series(pos.astype(float), index=candles.index), COST / 2)
            )
            for name, pos in positions.items()
        },
        orient="index",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp", action="store_true", help="보유 규칙을 GP로 탐색")
    parser.add_argument("--pop-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qqq, vix = load_yahoo_daily("QQQ"), load_yahoo_daily("^VIX")
    sessions = xc.get_calendar("XNYS", start="1999-01-01").sessions  # 기본값은 약 20년 전부터다
    X, y = features(qqq, vix, sessions), target(qqq)
    X, y = X[y.notna()], y[y.notna()]
    print(f"QQQ {X.index[0].date()} ~ {X.index[-1].date()} | 왕복 수수료 {COST:.2%}")

    if args.gp:
        table, hold = gp_search(X, y, args.pop_size, args.generations, args.seed)
        print(f"\n[GP 보유 규칙 (구간별 수수료 차감 Sharpe)]\n{table.round(2).to_string()}")
    else:
        hold = _hypothesis_hold(X, y)

    print(f"\n[QQQ ETF 밤 단위 백테스트 (수수료 차감, buy_and_hold는 수수료 없음)]\n{_etf_report(hold, y, qqq).round(4).to_string()}")
    perp = _perp_report(hold)
    print(f"\n[QQQUSDT 선물 적용 (편도 {COST / 2:.2%}, 연환산 단리)]\n{perp.round(4).to_string()}")


if __name__ == "__main__":
    main()
