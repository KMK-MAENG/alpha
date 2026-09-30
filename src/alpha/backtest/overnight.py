"""미국 정규장이 닫혀 있는 동안만 롱을 드는 오버나잇 전략을 1분봉으로 백테스트한다 (예: QQQUSDT TradFi 무기한 선물).

- overnight: 정규장 마감에 롱 진입 → 다음 정규장 개장에 청산. 주말·휴장일에도 계속 보유한다
- 비교: intraday(정규장 동안만 롱), buy_and_hold(계속 롱)
- 손익 = 가격 손익 + 펀딩(보유 중 정산 시각마다 -포지션 × 펀딩비) + 수수료(-|포지션 변화| × fee)
- 정규장 일정은 exchange_calendars의 NYSE(XNYS) 달력 (서머타임, 휴장일, 조기 폐장 반영)
- 수익률은 단리 합계로 집계한다 (구성 요소별로 나눠 보기 위해)

실행 (프로젝트 루트에서): uv run python -m alpha.backtest.overnight [--symbol QQQUSDT]
"""
import argparse

import exchange_calendars as xc
import numpy as np
import pandas as pd

from alpha.data import load_candles, load_funding

FEES = [0.0, 0.0002, 0.0005]  # 편도: 없음 / 메이커 / 테이커


def _ns(times) -> np.ndarray:
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def nyse_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    schedule = xc.get_calendar("XNYS").schedule
    return schedule.loc[start.tz_localize(None).normalize() : end.tz_localize(None) + pd.Timedelta(days=7)]


def _session_idx(index: pd.DatetimeIndex, schedule: pd.DataFrame) -> np.ndarray:
    """1분봉마다 직전(정규장 중이면 현재) 정규장의 schedule 행 번호. 첫 정규장 이전이면 -1."""
    return np.searchsorted(_ns(schedule["open"]), _ns(index), side="right") - 1


def session_dates(index: pd.DatetimeIndex, schedule: pd.DataFrame) -> pd.DatetimeIndex:
    """1분봉마다 직전(정규장 중이면 현재) 정규장 날짜. 장외 봉이면 그 구간이 시작된 마감의 날짜다."""
    return schedule.index[np.clip(_session_idx(index, schedule), 0, None)]


def label_sessions(index: pd.DatetimeIndex, schedule: pd.DataFrame) -> np.ndarray:
    """1분봉(open_time)마다 "rth"(정규장), "weeknight"(평일 밤), "weekend_holiday"(주말·휴장 포함 구간)."""
    t, opens, closes = _ns(index), _ns(schedule["open"]), _ns(schedule["close"])
    i = _session_idx(index, schedule)
    prev = np.clip(i, 0, None)
    rth = (i >= 0) & (t < closes[prev])
    gap = opens[np.clip(i + 1, 0, len(opens) - 1)] - closes[prev]  # 이 장외 구간의 길이
    weekend = gap >= pd.Timedelta(days=1).value
    return np.where(rth, "rth", np.where(weekend, "weekend_holiday", "weeknight"))


def backtest(candles: pd.DataFrame, funding: pd.Series, pos: pd.Series, fee: float) -> pd.DataFrame:
    """1분봉별 손익 구성. pos는 각 봉(open_time부터 1분) 동안의 포지션이다."""
    ret = candles["close"].pct_change().fillna(0)
    # 정산 시각이 속한 1분봉의 포지션에 펀딩을 매긴다
    rate = funding.groupby(funding.index.floor("min").as_unit(candles.index.unit)).sum()
    rate = rate.reindex(candles.index, fill_value=0.0)
    pnl = pd.DataFrame(
        {"price": pos * ret, "funding": -pos * rate, "fee": -fee * pos.diff().abs().fillna(pos.abs())}
    )
    pnl["total"] = pnl.sum(axis=1)
    return pnl


def _years(index: pd.DatetimeIndex) -> float:
    return (index[-1] - index[0]) / pd.Timedelta(days=365)


def summary(pnl: pd.DataFrame) -> dict:
    daily = pnl["total"].resample("1D").sum()
    years = _years(pnl.index)
    equity = daily.cumsum()
    return {
        "연수익": pnl["total"].sum() / years,
        "가격": pnl["price"].sum() / years,
        "펀딩": pnl["funding"].sum() / years,
        "수수료": pnl["fee"].sum() / years,
        "연변동성": daily.std() * np.sqrt(365),
        "Sharpe": daily.mean() / daily.std() * np.sqrt(365),
        "MDD": (equity - equity.cummax()).min(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="QQQUSDT")
    args = parser.parse_args()

    candles = load_candles("1m", args.symbol)
    funding = load_funding(args.symbol).loc[candles.index[0] :]
    labels = label_sessions(candles.index, nyse_schedule(candles.index[0], candles.index[-1]))
    positions = {
        "overnight": pd.Series((labels != "rth").astype(float), index=candles.index),
        "intraday": pd.Series((labels == "rth").astype(float), index=candles.index),
        "buy_and_hold": pd.Series(1.0, index=candles.index),
    }

    days = (candles.index[-1] - candles.index[0]).days
    print(f"{args.symbol} 1분봉 {candles.index[0]} ~ {candles.index[-1]} ({days}일, 단리 기준, 연환산)")
    rows = {
        (name, f"{fee:.2%}"): summary(backtest(candles, funding, pos, fee))
        for name, pos in positions.items()
        for fee in FEES
    }
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis(["전략", "수수료(편도)"])
    pct = ["연수익", "가격", "펀딩", "수수료", "연변동성", "MDD"]
    print(table.round(2).astype({"Sharpe": str}).assign(**{c: table[c].map("{:.2%}".format) for c in pct}).to_string())

    pnl = backtest(candles, funding, positions["overnight"], 0.0)
    years = _years(candles.index)
    print("\n[오버나잇 가격 손익이 어디서 나오나 (연환산)]")
    for label in ["weeknight", "weekend_holiday", "rth"]:
        mask = labels == label
        print(f"  {label:16}: 가격 {pnl['price'][mask].sum() / years:+.2%} | 기초자산 움직임 {candles['close'].pct_change()[mask].sum() / years:+.2%}")


if __name__ == "__main__":
    main()
