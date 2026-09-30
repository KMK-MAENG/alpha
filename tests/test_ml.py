import numpy as np
import pandas as pd
import pytest

from alpha.strategies import ml


def fake_candles(n=500, seed=0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="1D", tz="UTC", name="open_time")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))
    volume = rng.uniform(1e3, 2e3, n)
    return pd.DataFrame(
        {"close": close, "volume": volume, "quote_volume": volume * close, "taker_buy_volume": volume * 0.5},
        index=index,
    )


def fake_funding(candles):
    # 마지막 날의 00·08·16시 정산까지 (그날 종가 시점에 모두 알려져 있다)
    times = pd.date_range(candles.index[0], candles.index[-1] + pd.Timedelta(hours=16), freq="8h")
    return pd.Series(0.0001, index=times)


def test_features_use_only_past_data():
    candles = fake_candles()
    full = ml.coin_features(candles, fake_funding(candles)).iloc[:300]
    head = ml.coin_features(candles.iloc[:300], fake_funding(candles.iloc[:300]))
    pd.testing.assert_frame_equal(full, head)


def test_target_is_future_return_scaled_by_volatility():
    candles = fake_candles()
    y = ml.target(candles["close"])
    c = candles["close"]
    vol = c.pct_change().rolling(60).std()
    t = 100
    expected = (c.iloc[t + ml.HORIZON] / c.iloc[t] - 1) / (vol.iloc[t] * np.sqrt(ml.HORIZON))
    assert y.iloc[t] == pytest.approx(np.clip(expected, -3, 3))
    assert y.iloc[-ml.HORIZON:].isna().all()


class SpyModel:
    """학습에 쓰인 마지막 날짜를 기록하는 가짜 모델."""

    fits = []

    def fit(self, X, y):
        SpyModel.fits.append(X.index.get_level_values("date").max())
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_walk_forward_trains_only_on_labels_known_before_each_prediction():
    candles = {f"C{i}": fake_candles(900, seed=i) for i in range(3)}
    panel = ml.panel(candles, {s: fake_funding(c) for s, c in candles.items()}, market="C0")
    SpyModel.fits = []
    preds = ml.walk_forward_predictions(panel, model_factory=SpyModel)
    first_pred = preds.index.get_level_values("date").min()
    assert first_pred >= panel.index.get_level_values("date").min() + pd.Timedelta(days=ml.MIN_TRAIN_DAYS)
    starts = sorted(set(preds.index.get_level_values("date").tz_localize(None).to_period("Q").start_time.tz_localize("UTC")))
    assert len(SpyModel.fits) == len(starts)
    for fit_end, start in zip(SpyModel.fits, starts):
        # 라벨은 HORIZON일 뒤 종가를 쓰므로, 예측 시작 시점에 알 수 있는 라벨만 학습해야 한다
        assert fit_end <= start - pd.Timedelta(days=ml.HORIZON)
