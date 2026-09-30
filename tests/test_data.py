import pandas as pd
import pytest

from alpha.data import add_derivatives, load_candles


def test_load_candles_indexes_by_open_time(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_DATA_DIR", str(tmp_path))
    pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"], utc=True),
            "open": [1.0, 2.0],
            "close": [2.0, 3.0],
        }
    ).to_parquet(tmp_path / "btcusdt_1h.parquet", index=False)

    df = load_candles("1h")

    assert df.index.name == "open_time"
    assert str(df.index.tz) == "UTC"
    assert list(df.columns) == ["open", "close"]
    assert df.loc["2024-01-01 01:00", "open"] == 2.0


def test_load_candles_raises_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_candles("1h")


def test_add_derivatives_uses_only_values_published_before_bar_close(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_DATA_DIR", str(tmp_path))
    t = lambda *xs: pd.to_datetime(list(xs), utc=True)
    pd.DataFrame({"funding_time": t("2024-01-01 00:00", "2024-01-01 08:00"), "funding_rate": [0.2, 0.1]}).to_parquet(
        tmp_path / "btcusdt_funding.parquet", index=False
    )
    pd.DataFrame(
        {
            "create_time": t("2024-01-01 07:55", "2024-01-01 08:00"),
            "sum_open_interest": [100.0, 200.0],
            "count_long_short_ratio": [1.5, 2.5],
        }
    ).to_parquet(tmp_path / "btcusdt_metrics.parquet", index=False)
    candles = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]}, index=pd.Index(t("2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"), name="open_time")
    )

    df = add_derivatives(candles, "4h")

    # 04:00 봉은 08:00에 마감하므로 08:00 정각에 공개된 값은 쓰지 않는다
    assert df["funding_rate"].tolist() == [0.2, 0.2, 0.1]
    assert df["open_interest"].tolist()[1:] == [100.0, 200.0]
    assert df["ls_account"].tolist()[1:] == [1.5, 2.5]
    assert pd.isna(df["open_interest"].iloc[0])
