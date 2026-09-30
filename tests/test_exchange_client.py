import pandas as pd
import pytest
from binance.exceptions import BinanceAPIException

from alpha.exchange.client import BinanceFutures
from alpha.execution.order_manager import Order, SymbolRules

DAY_MS = 86_400_000
NOW = pd.Timestamp("2026-09-29 00:05", tz="UTC")


def kline(day: str, close: float) -> list:
    t = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    return [t, "1", "1", "1", str(close), "1", t + DAY_MS - 1, "1", 1, "1", "1", "0"]


class FakeClient:
    def __init__(self):
        self.orders, self.calls = [], []

    def futures_klines(self, symbol, interval, limit):
        assert interval == "1d"
        return [kline("2026-09-27", 10.0), kline("2026-09-28", 11.0), kline("2026-09-29", 12.0)]  # 마지막은 진행 중

    def futures_exchange_info(self):
        def sym(name, status="TRADING", contract="PERPETUAL"):
            return {
                "symbol": name, "status": status, "contractType": contract,
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        return {"symbols": [sym("AAAUSDT"), sym("BBBUSDT", status="SETTLING"), sym("CCCUSDT", contract="CURRENT_QUARTER")]}

    def futures_mark_price(self):
        return [{"symbol": "AAAUSDT", "markPrice": "101.5"}]

    def futures_account(self):
        return {"totalMarginBalance": "2000.5", "totalWalletBalance": "1990.0", "totalUnrealizedProfit": "10.5",
                "availableBalance": "1500.0"}

    def futures_position_information(self):
        return [
            {"symbol": "AAAUSDT", "positionAmt": "0.300", "entryPrice": "100.0", "unRealizedProfit": "0.45"},
            {"symbol": "BBBUSDT", "positionAmt": "0.000", "entryPrice": "0.0", "unRealizedProfit": "0.0"},
        ]

    def futures_income_history(self, incomeType, startTime, endTime, limit):
        ms = lambda t: int(pd.Timestamp(t, tz="UTC").timestamp() * 1000)
        rows = [
            {"incomeType": "TRANSFER", "asset": "USDT", "income": "1000.0", "time": ms("2026-09-29 12:00")},
            {"incomeType": "TRANSFER", "asset": "USDT", "income": "-300.0", "time": ms("2026-09-30 03:00")},
            {"incomeType": "TRANSFER", "asset": "BNB", "income": "5.0", "time": ms("2026-09-29 13:00")},
            {"incomeType": "TRANSFER", "asset": "USDT", "income": "50.0", "time": ms("2026-10-02 00:00")},  # 기간 밖
        ]
        return [r for r in rows if r["incomeType"] == incomeType and startTime <= r["time"] <= endTime]

    def futures_get_position_mode(self):
        return {"dualSidePosition": False}

    def futures_create_order(self, **params):
        self.orders.append(params)
        return {"orderId": 1, "status": "FILLED", **params}

    def futures_change_margin_type(self, symbol, marginType):
        self.calls.append(("margin", symbol, marginType))
        response = type("R", (), {"status_code": 400, "text": '{"code":-4046,"msg":"No need to change margin type."}'})
        raise BinanceAPIException(response, 400, response.text)

    def futures_change_leverage(self, symbol, leverage):
        self.calls.append(("leverage", symbol, leverage))


@pytest.fixture
def exchange():
    return BinanceFutures(client=FakeClient())


def test_daily_closes_excludes_unfinished_candle(exchange):
    closes = exchange.daily_closes(["AAAUSDT"], days=300, now=NOW)
    assert closes["AAAUSDT"].tolist() == [10.0, 11.0]
    assert closes.index[-1] == pd.Timestamp("2026-09-28", tz="UTC")


def test_rules_only_for_trading_perpetuals(exchange):
    assert exchange.rules() == {"AAAUSDT": SymbolRules(min_notional=5.0, min_qty=0.001, step=0.001)}


def test_account_state(exchange):
    assert exchange.equity() == 2000.5
    assert exchange.balances() == {"equity": 2000.5, "wallet": 1990.0, "unrealized": 10.5, "available": 1500.0}
    assert exchange.positions().to_dict() == {"AAAUSDT": 0.3}
    details = exchange.position_details()
    assert details.loc["AAAUSDT"].to_dict() == {"qty": 0.3, "entry_price": 100.0, "unrealized_pnl": 0.45}
    assert exchange.mark_prices().to_dict() == {"AAAUSDT": 101.5}
    assert exchange.is_one_way()


def test_market_order_formats_quantity_and_reduce_only(exchange):
    exchange.market_order(Order("AAAUSDT", "SELL", 0.2, reduce_only=True, notional=20.0))
    exchange.market_order(Order("AAAUSDT", "BUY", 1.0, reduce_only=False, notional=100.0))
    first, second = exchange._client.orders
    assert first == {"symbol": "AAAUSDT", "side": "SELL", "type": "MARKET", "quantity": "0.2", "reduceOnly": "true",
                     "newOrderRespType": "RESULT"}
    assert second["quantity"] == "1" and "reduceOnly" not in second


def test_setup_ignores_already_set_margin_type(exchange):
    exchange.setup(["AAAUSDT"], leverage=3)
    assert exchange._client.calls == [("margin", "AAAUSDT", "CROSSED"), ("leverage", "AAAUSDT", 3)]


def test_net_transfers_sums_usdt_transfers_in_period(exchange):
    start, end = pd.Timestamp("2026-09-29", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
    assert exchange.net_transfers(start, end) == pytest.approx(700.0)
