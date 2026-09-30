from pathlib import Path

import pandas as pd

from alpha.config import get_env


def _read(name: str, symbol: str) -> pd.DataFrame:
    return pd.read_parquet(Path(get_env("CRYPTO_DATA_DIR")) / f"{symbol.lower()}_{name}.parquet")


def load_candles(interval: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    return _read(interval, symbol).set_index("open_time")


def vision_symbols() -> list[str]:
    """update.py --vision으로 받은 심볼 (상장 폐지 코인 포함)."""
    folder = Path(get_env("CRYPTO_DATA_DIR")) / "vision"
    return sorted(p.name.removesuffix("_1d.parquet").upper() for p in folder.glob("*_1d.parquet"))


def load_vision(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    """data.binance.vision 일봉과 펀딩비 (지난달까지). 펀딩 파일이 없으면 빈 시계열.
    주의: 상장 폐지 뒤에도 가격 고정·거래량 0인 채움 기록이 이어지는 심볼이 많다 (volume == 0인 날은 거래 없음)."""
    folder = Path(get_env("CRYPTO_DATA_DIR")) / "vision"
    candles = pd.read_parquet(folder / f"{symbol.lower()}_1d.parquet").set_index("open_time")
    path = folder / f"{symbol.lower()}_funding.parquet"
    if path.exists():
        funding = pd.read_parquet(path).set_index("funding_time")["funding_rate"]
    else:
        funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    return candles, funding


def load_yahoo_daily(ticker: str) -> pd.DataFrame:
    """Yahoo Finance 일봉 (배당·분할 조정 OHLCV). 캐시 없이 매번 받는다."""
    import yfinance as yf

    df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False, multi_level_index=False)
    return df.rename(columns=str.lower).rename_axis("date")


def load_funding(symbol: str = "BTCUSDT") -> pd.Series:
    return _read("funding", symbol).set_index("funding_time")["funding_rate"]


def add_derivatives(candles: pd.DataFrame, interval: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """캔들에 펀딩비(funding_rate), 미결제약정(open_interest), 전체 계정 롱/숏 비율(ls_account)을 붙인다.

    각 봉에는 봉 마감 시각 '이전'에 공개된 마지막 값만 쓴다. 펀딩 정산 시각(00/08/16시)이 봉 경계와 겹쳐도
    마감과 같은 시각의 값은 다음 봉부터 반영되므로 미래 참조가 없다. 파생 지표는 2020-09부터 있다.
    """
    funding = _read("funding", symbol).rename(columns={"funding_time": "time"})
    metrics = _read("metrics", symbol).rename(
        columns={"create_time": "time", "sum_open_interest": "open_interest", "count_long_short_ratio": "ls_account"}
    )[["time", "open_interest", "ls_account"]]

    close_time = pd.DataFrame({"close_time": (candles.index + pd.Timedelta(interval)).astype("datetime64[ns, UTC]")})
    out = candles.copy()
    for source in (funding, metrics):
        source = source.astype({"time": "datetime64[ns, UTC]"}).sort_values("time")
        merged = pd.merge_asof(close_time, source, left_on="close_time", right_on="time", allow_exact_matches=False)
        for column in source.columns.drop("time"):
            out[column] = merged[column].to_numpy()
    return out
