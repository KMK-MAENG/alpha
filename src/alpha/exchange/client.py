"""바이낸스 USDT-M 선물 통신만 담당한다 (python-binance 래퍼). 주문 결정은 하지 않는다."""
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from alpha.execution.order_manager import Order, SymbolRules

NO_NEED_TO_CHANGE_MARGIN_TYPE = -4046


class BinanceFutures:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None, client=None):
        self._client = client or Client(api_key, api_secret)

    def daily_closes(self, symbols: list[str], days: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
        """마감된 일봉 종가 (날짜 × 코인). 진행 중인 오늘 봉은 제외한다."""
        today = (now or pd.Timestamp.now(tz="UTC")).floor("D")
        closes = {}
        for symbol in symbols:
            rows = self._client.futures_klines(symbol=symbol, interval="1d", limit=days + 1)
            index = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
            series = pd.Series([float(r[4]) for r in rows], index=index)
            closes[symbol] = series[series.index < today]
        return pd.DataFrame(closes)

    def rules(self) -> dict[str, SymbolRules]:
        """거래 중인 무기한 선물의 주문 규칙."""
        out = {}
        for s in self._client.futures_exchange_info()["symbols"]:
            if s["status"] != "TRADING" or s["contractType"] != "PERPETUAL":
                continue
            f = {x["filterType"]: x for x in s["filters"]}
            out[s["symbol"]] = SymbolRules(
                min_notional=float(f["MIN_NOTIONAL"]["notional"]),
                min_qty=float(f["LOT_SIZE"]["minQty"]),
                step=float(f["LOT_SIZE"]["stepSize"]),
            )
        return out

    def mark_prices(self) -> pd.Series:
        return pd.Series({p["symbol"]: float(p["markPrice"]) for p in self._client.futures_mark_price()})

    def equity(self) -> float:
        """계좌 평가금액 = 지갑 잔고 + 미실현 손익."""
        return self.balances()["equity"]

    def balances(self) -> dict[str, float]:
        a = self._client.futures_account()
        return {
            "equity": float(a["totalMarginBalance"]),
            "wallet": float(a["totalWalletBalance"]),
            "unrealized": float(a["totalUnrealizedProfit"]),
            "available": float(a["availableBalance"]),
        }

    def net_transfers(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """기간 안의 선물 지갑 USDT 순입금액 (입금 +, 출금 -)."""
        start_ms, end_ms, total = int(start.timestamp() * 1000), int(end.timestamp() * 1000), 0.0
        while True:
            rows = self._client.futures_income_history(incomeType="TRANSFER", startTime=start_ms, endTime=end_ms, limit=1000)
            total += sum(float(r["income"]) for r in rows if r["asset"] == "USDT")
            if len(rows) < 1000:
                return total
            start_ms = rows[-1]["time"] + 1

    def position_details(self) -> pd.DataFrame:
        """0이 아닌 포지션의 수량(롱 +, 숏 -)·진입가·미실현 손익. 단방향 모드 기준."""
        rows = {
            p["symbol"]: {
                "qty": float(p["positionAmt"]),
                "entry_price": float(p["entryPrice"]),
                "unrealized_pnl": float(p["unRealizedProfit"]),
            }
            for p in self._client.futures_position_information()
            if float(p["positionAmt"]) != 0
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=["qty", "entry_price", "unrealized_pnl"], dtype=float)

    def positions(self) -> pd.Series:
        """0이 아닌 포지션 수량 (롱 +, 숏 -)."""
        return self.position_details()["qty"]

    def is_one_way(self) -> bool:
        return not self._client.futures_get_position_mode()["dualSidePosition"]

    def market_order(self, order: Order) -> dict:
        params = {
            "symbol": order.symbol,
            "side": order.side,
            "type": "MARKET",
            "quantity": f"{order.quantity:.10f}".rstrip("0").rstrip("."),
            "newOrderRespType": "RESULT",
        }
        if order.reduce_only:
            params["reduceOnly"] = "true"
        return self._client.futures_create_order(**params)

    def setup(self, symbols: list[str], leverage: int) -> None:
        """교차 증거금·레버리지 설정 (이미 설정돼 있으면 그대로)."""
        for symbol in symbols:
            try:
                self._client.futures_change_margin_type(symbol=symbol, marginType="CROSSED")
            except BinanceAPIException as e:
                if e.code != NO_NEED_TO_CHANGE_MARGIN_TYPE:
                    raise
            self._client.futures_change_leverage(symbol=symbol, leverage=leverage)
