"""XGBoost 신호: 여러 코인을 모은 패널로 '향후 HORIZON일 수익률 ÷ 변동성'을 예측하고, 예측의 부호를 신호로 쓴다.

- 피처는 모두 날짜 t 종가까지의 정보이고 코인 간 비교가 되도록 척도를 맞춘다 (수익률·이동평균 괴리는 변동성으로 나눔)
- 타깃: t+1 ~ t+HORIZON 수익률 ÷ (60일 변동성 × √HORIZON), ±3에서 자름
- 워크포워드: 첫 MIN_TRAIN_DAYS 뒤 분기 시작마다 그때까지 알 수 있는 라벨로 재학습하고 다음 분기를 예측한다.
  라벨은 HORIZON일 뒤 종가를 쓰므로 예측 시작일보다 HORIZON일 이상 앞선 행만 학습한다 (미래 누설 방지)
- 하이퍼파라미터는 실행 전에 고정했다 (튜닝 없음)
"""
from typing import Callable

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from alpha.strategies.indicators import _percent_b, ema, rsi, sma

HORIZON = 5
MIN_TRAIN_DAYS = 365
PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=50, random_state=0, n_jobs=4,
)


def coin_features(candles: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    c = candles["close"]
    r = c.pct_change()
    vol = r.rolling(60).std()
    f = {f"ret_{n}": c.pct_change(n) / (vol * np.sqrt(n)) for n in (1, 5, 10, 20, 60, 120)}
    f |= {f"sma_gap_{n}": (c / sma(c, n) - 1) / vol for n in (20, 60, 120, 200)}
    macd_line = ema(c, 12) - ema(c, 26)
    hi, lo = c.rolling(55).max(), c.rolling(55).min()
    daily_funding = funding.resample("1D").sum()  # 정산일(00·08·16시) 기준 — 그날 종가 시점에 모두 알려져 있다
    daily_funding.index = daily_funding.index.as_unit(c.index.unit)
    f |= {
        "rsi14": rsi(c) / 100,
        "bb_percent_b": _percent_b(c),
        "macd_hist": (macd_line - ema(macd_line, 9)) / (c * vol),
        "donchian_pos": (c - lo) / (hi - lo),
        "vol_60": vol * np.sqrt(365),
        "vol_ratio": r.rolling(20).std() / vol,
        "volume_surge": candles["quote_volume"] / candles["quote_volume"].rolling(20).mean(),
        "taker_ratio_20": (candles["taker_buy_volume"] / candles["volume"]).rolling(20).mean(),
        "funding_7d": daily_funding.reindex(c.index).rolling(7).mean(),
    }
    return pd.DataFrame(f, index=c.index).replace([np.inf, -np.inf], np.nan)


def target(close: pd.Series) -> pd.Series:
    vol = close.pct_change().rolling(60).std()
    forward = close.shift(-HORIZON) / close - 1
    return (forward / (vol * np.sqrt(HORIZON))).clip(-3, 3)


def panel(candles: dict[str, pd.DataFrame], fundings: dict[str, pd.Series], market: str = "BTCUSDT") -> pd.DataFrame:
    """(date, symbol) 행의 피처 + 시장(market 코인) 추세 피처 + target."""
    feats = {s: coin_features(c, fundings[s]) for s, c in candles.items()}
    market_feats = feats[market][["ret_20", "ret_60"]].add_prefix("market_")
    frames = {s: f.join(market_feats).assign(target=target(candles[s]["close"])) for s, f in feats.items()}
    out = pd.concat(frames, names=["symbol", "date"]).swaplevel().sort_index()
    return out.dropna(subset=["ret_20"])  # 기본 추세 피처도 없는 초기 행 제외


def walk_forward_predictions(
    data: pd.DataFrame, model_factory: Callable = lambda: XGBRegressor(**PARAMS)
) -> pd.Series:
    features = [c for c in data.columns if c != "target"]
    dates = data.index.get_level_values("date")
    first = dates.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    quarter = first.tz_localize(None).to_period("Q").start_time  # 분기 시작일 (UTC 기준)
    starts = pd.date_range(quarter, dates.max().tz_localize(None), freq="QS").tz_localize("UTC")
    starts = [s for s in starts if s >= first] or [first]
    preds = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else dates.max() + pd.Timedelta(days=1)
        train = data[(dates <= start - pd.Timedelta(days=HORIZON)) & data["target"].notna()]
        test = data[(dates >= start) & (dates < end)]
        if test.empty:
            continue
        model = model_factory().fit(train[features], train["target"])
        preds.append(pd.Series(model.predict(test[features]), index=test.index))
    return pd.concat(preds)
