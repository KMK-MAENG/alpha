"""WorldQuant "101 Formulaic Alphas"(Kakushadze, 2015, arXiv:1601.00991)를 BTCUSDT 1h용 gplearn 시드와 연산자로 옮긴다.

단일 종목에서는 횡단면 연산이 무의미하므로 다음처럼 변환한다.
- rank(x) → ts_rank(x, 24): 다른 종목 대비 순위 대신 자기 최근 24봉 대비 순위 (0~1 스케일 유지)
- scale(x), indneutralize(x, g) → x
- returns → ret, adv{d} → ts_mean(volume, d): 논문 정의는 달러 거래대금이지만 volume / adv20처럼
  거래량과 직접 비교하는 수식이 많아 거래량 평균으로 둔다
- 윈도우는 일수를 1h 봉 수로 그대로 읽고 소수는 버린 뒤(논문 규칙), 허용 윈도우 중 로그 거리로 가장 가까운 값에 맞춘다.
  윈도우 1인 sum/min/max 등은 x 자체다
- a == b는 실수 비교라 사실상 항상 거짓이므로 0으로 둔다
- sum(x, d) → d * ts_mean(x, w): 윈도우를 w로 맞춰도 sum(x, d) / d(= 평균) 같은 비교의 크기가 유지된다
- cap(시가총액)이 필요한 #56, 긴 윈도우 product를 쓰는 #81은 제외한다

실행 (프로젝트 루트에서):
  uv run python -m alpha.mining.wq101 [--interval 4h] [--start 2020-09-01] [--derivatives] [--pop-size N] [--generations N] [--seed N]
"""
import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.functions import make_function
from numpy.lib.stride_tricks import sliding_window_view

from alpha.data import add_derivatives, load_candles
from alpha.mining import expr, gplearn_search, search

RANK_WINDOW = 24
DELTA_WINDOWS = [1, 2, *expr.WINDOWS]  # delay/delta는 1봉 차이가 핵심이라 1, 2도 허용


def _rolling_window(x: pd.Series, d: int, fn) -> pd.Series:
    """길이 d 윈도우마다 fn(2차원 배열) → 1차원. 윈도우에 NaN이 있으면 NaN."""
    out = np.full(len(x), np.nan)
    if len(x) >= d:
        windows = sliding_window_view(x.to_numpy(dtype="float64"), d)
        out[d - 1 :] = np.where(np.isnan(windows).any(axis=1), np.nan, fn(windows))
    return pd.Series(out, index=x.index)


def _decay_linear(x, d):
    weights = np.arange(1, d + 1) / (d * (d + 1) / 2)  # 가장 최근 값에 가중치 d
    return _rolling_window(x, d, lambda w: w @ weights)


# expr.OPS와 같은 형식: 이름: (함수, 수식 인자 수, 윈도우 인자 여부)
OPS = {
    "delay": (lambda x, d: x.shift(d), 1, True),
    "decay_linear": (_decay_linear, 1, True),
    "ts_argmax": (lambda x, d: _rolling_window(x, d, lambda w: w.argmax(axis=1)), 1, True),
    "ts_argmin": (lambda x, d: _rolling_window(x, d, lambda w: w.argmin(axis=1)), 1, True),
    "ts_cov": (lambda a, b, d: a.rolling(d).cov(b), 2, True),
}


def _spow(a, b):
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.sign(a) * np.abs(a) ** b, nan=0.0, posinf=0.0, neginf=0.0)


FUNCTIONS = (
    gplearn_search.FUNCTIONS
    + ["max", "min"]
    + [
        make_function(function=lambda x: np.sign(x), name="sign", arity=1, wrap=False),
        make_function(function=_spow, name="spow", arity=2, wrap=False),
        make_function(function=lambda a, b: (a < b).astype("float64"), name="lt", arity=2, wrap=False),
        make_function(function=lambda c, a, b: np.where(c > 0, a, b), name="where", arity=3, wrap=False),
    ]
    + [gplearn_search.ts_function("ts_delta", expr.OPS["ts_delta"][0], 1, d) for d in (1, 2)]
    + [
        gplearn_search.ts_function(name, fn, arity, d)
        for name, (fn, arity, _) in OPS.items()
        for d in (DELTA_WINDOWS if name == "delay" else expr.WINDOWS)
    ]
)


class Unsupported(ValueError):
    pass


# 논문 함수 → 시계열 연산자 (마지막 인자가 윈도우)
_TS_NAMES = {
    "delay": "delay", "delta": "ts_delta", "correlation": "ts_corr", "covariance": "ts_cov",
    "decay_linear": "decay_linear", "ts_min": "ts_min", "ts_max": "ts_max", "min": "ts_min", "max": "ts_max",
    "ts_argmax": "ts_argmax", "ts_argmin": "ts_argmin", "ts_rank": "ts_rank", "sum": "ts_sum",
    "product": "ts_product", "stddev": "ts_std",
}
_BINOPS = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "spow", "<": "lt", "||": "max"}
_FOLD = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b,
    "^": lambda a, b: a**b, "<": lambda a, b: float(a < b), "||": lambda a, b: float(max(a, b) > 0),
}
_PREC = {"?": 1, "||": 2, "==": 3, "<": 4, ">": 4, "+": 5, "-": 5, "*": 6, "/": 6, "^": 7}
_TOKEN = re.compile(r"\s*(\d+\.?\d*|\.\d+|[A-Za-z_][\w.]*|\|\||==|[-+*/^<>?:(),])")


def _ts(op: str, args: list, d: float):
    d = math.floor(d)
    if d <= 1 and op not in ("delay", "ts_delta"):
        return args[0]
    if op == "ts_product":
        raise Unsupported("product")
    windows = DELTA_WINDOWS if op in ("delay", "ts_delta") else expr.WINDOWS
    w = min(windows, key=lambda w: abs(math.log(w / d)))
    if op == "ts_sum":
        # 윈도우를 맞춰도 합계 크기가 유지되도록 d * 평균으로 둔다 (논문은 평균을 sum(x, d) / d로 쓴다)
        return ("mul", float(d), (f"ts_mean_{w}", *args))
    return (f"{op}_{w}", *args)


def _binop(op: str, a, b):
    if op == ">":
        return _binop("<", b, a)
    if op == "==":
        return 0.0
    if isinstance(a, float) and isinstance(b, float):
        return _FOLD[op](a, b)
    if op == "*" and (a == -1.0 or b == -1.0):
        return ("neg", b if a == -1.0 else a)
    if op == "*" and a == 1.0:
        return b
    if (op in ("*", "/", "^") and b == 1.0) or (op in ("+", "-", "||") and b == 0.0):
        return a
    if op == "-" and a == 0.0:
        return ("neg", b)
    if op == "/" and isinstance(a, tuple) and a[0] == "mul" and a[1] == b:
        return a[2]  # (d * x) / d → x
    return (_BINOPS[op], a, b)


class _Parser:
    def __init__(self, text: str):
        self.tokens = _TOKEN.findall(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected=None):
        tok = self.peek()
        if expected is not None and tok != expected:
            raise SyntaxError(f"expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self, min_prec: int = 0):
        left = self.unary()
        while (op := self.peek()) in _PREC and _PREC[op] > min_prec:
            self.take()
            if op == "?":
                a = self.parse()
                self.take(":")
                left = ("where", left, a, self.parse())
            else:
                right = self.parse(_PREC[op] - 1 if op == "^" else _PREC[op])
                left = _binop(op, left, right)
        return left

    def unary(self):
        if self.peek() == "-":
            self.take()
            x = self.unary()
            return -x if isinstance(x, float) else ("neg", x)
        return self.atom()

    def atom(self):
        tok = self.take()
        if tok == "(":
            x = self.parse()
            self.take(")")
            return x
        if tok[0].isdigit() or tok[0] == ".":
            return float(tok)
        if self.peek() == "(":
            self.take("(")
            args = [self.parse()]
            while self.peek() == ",":
                self.take()
                args.append(self.parse())
            self.take(")")
            return _call(tok.lower(), args)
        return _variable(tok.lower())


def _call(name: str, args: list):
    if name in ("min", "max") and not isinstance(args[1], float):
        return (name, *args)  # 두 번째 인자가 수식이면 원소별 min/max
    if name in _TS_NAMES:
        return _ts(_TS_NAMES[name], args[:-1], args[-1])
    if name == "rank":
        return _ts("ts_rank", args, RANK_WINDOW)
    if name in ("scale", "indneutralize"):
        return args[0]
    if name == "signedpower":
        return _binop("^", *args)
    if name in ("abs", "log", "sign"):
        return (name, args[0])
    raise Unsupported(name)


def _variable(name: str):
    if name == "returns":
        return "ret"
    if name in ("open", "high", "low", "close", "volume", "vwap"):
        return name
    if m := re.fullmatch(r"adv(\d+)", name):
        return _ts("ts_mean", ["volume"], int(m.group(1)))
    if name.startswith("indclass."):
        return name  # indneutralize의 그룹 인자, 버려진다
    raise Unsupported(name)


def translate(formula: str):
    """논문 수식 → 함수 이름 튜플 수식 (gplearn_search.execute / 시드용)."""
    return _Parser(formula).parse()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", help="이 날짜 이후 데이터만 사용 (예: 2020-09-01)")
    parser.add_argument("--derivatives", action="store_true", help="펀딩비·미결제약정·롱숏 비율 피처 추가")
    parser.add_argument("--pop-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    search.set_interval(args.interval)
    candles = load_candles(args.interval)
    if args.derivatives:
        candles = add_derivatives(candles, args.interval)
    candles = candles.loc[args.start :]
    X, ret_next = gplearn_search.inputs(candles)
    originals = {f"wq#{n}": gplearn_search.execute(e, X, FUNCTIONS) for n, e in SEEDS.items()}
    baseline = search.report(search.select(originals, ret_next), ret_next)
    evolved = gplearn_search.run(candles, args.pop_size, args.generations, args.seed, FUNCTIONS, SEEDS)

    tag = f"{args.interval}{'_deriv' if args.derivatives else ''}"
    stamp = f"{tag}_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}"
    for name, result in [("wq101_original", baseline), ("wq101_evolved", evolved)]:
        print(f"\n===== {name}\n{result.round(2).to_string()}")
        out = Path("data") / f"{name}_{stamp}.csv"
        result.to_csv(out)
        print(f"저장: {out}")


def _translatable(formulas: dict[int, str]) -> dict:
    seeds = {}
    for n, formula in formulas.items():
        try:
            seeds[n] = translate(formula)
        except Unsupported:
            pass
    return seeds


# 논문 부록 A.1의 수식 원문
FORMULAS = {
    1: '(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)',
    2: '(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))',
    3: '(-1 * correlation(rank(open), rank(volume), 10))',
    4: '(-1 * Ts_Rank(rank(low), 9))',
    5: '(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))',
    6: '(-1 * correlation(open, volume, 10))',
    7: '((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))',
    8: '(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))',
    9: '((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))',
    10: 'rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))',
    11: '((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))',
    12: '(sign(delta(volume, 1)) * (-1 * delta(close, 1)))',
    13: '(-1 * rank(covariance(rank(close), rank(volume), 5)))',
    14: '((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))',
    15: '(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))',
    16: '(-1 * rank(covariance(rank(high), rank(volume), 5)))',
    17: '(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))',
    18: '(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))',
    19: '((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))',
    20: '(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))',
    21: '((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))',
    22: '(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))',
    23: '(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)',
    24: '((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))',
    25: 'rank(((((-1 * returns) * adv20) * vwap) * (high - close)))',
    26: '(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))',
    27: '((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)',
    28: 'scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))',
    29: '(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))',
    30: '(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))',
    31: '((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))',
    32: '(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))',
    33: 'rank((-1 * ((1 - (open / close))^1)))',
    34: 'rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))',
    35: '((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))',
    36: '(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))',
    37: '(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))',
    38: '((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))',
    39: '((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))',
    40: '((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))',
    41: '(((high * low)^0.5) - vwap)',
    42: '(rank((vwap - close)) / rank((vwap + close)))',
    43: '(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))',
    44: '(-1 * correlation(high, rank(volume), 5))',
    45: '(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))',
    46: '((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))',
    47: '((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))',
    48: '(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))',
    49: '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    50: '(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))',
    51: '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    52: '((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))',
    53: '(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))',
    54: '((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))',
    55: '(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))',
    56: '(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))',
    57: '(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))',
    58: '(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))',
    59: '(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))',
    60: '(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))',
    61: '(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))',
    62: '((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)',
    63: '((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)',
    64: '((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)',
    65: '((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)',
    66: '((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)',
    67: '((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)',
    68: '((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)',
    69: '((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)',
    70: '((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)',
    71: 'max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))',
    72: '(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))',
    73: '(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)',
    74: '((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)',
    75: '(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))',
    76: '(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)',
    77: 'min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))',
    78: '(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))',
    79: '(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))',
    80: '((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)',
    81: '((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)',
    82: '(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)',
    83: '((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))',
    84: 'SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))',
    85: '(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))',
    86: '((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)',
    87: '(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)',
    88: 'min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))',
    89: '(Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))',
    90: '((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)',
    91: '((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)',
    92: 'min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))',
    93: '(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))',
    94: '((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)',
    95: '(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))',
    96: '(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)',
    97: '((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)',
    98: '(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))',
    99: '((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)',
    100: '(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))',
    101: '((close - open) / ((high - low) + .001))',
}

SEEDS = _translatable(FORMULAS)


if __name__ == "__main__":
    main()
