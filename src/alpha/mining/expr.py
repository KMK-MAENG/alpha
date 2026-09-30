"""알파 수식 트리: 입력 피처, 연산자, 랜덤 생성/변이/교차.

수식은 중첩 튜플이다. 피처는 문자열("close"), 연산은 (연산자, *수식 인자, [윈도우]).
예: ("ts_mean", ("ts_delta", "close", 6), 24) == ts_mean(ts_delta(close, 6), 24)
모든 연산자는 과거 데이터만 사용한다 (미래 참조 없음).
"""
import random

import numpy as np
import pandas as pd

MAX_DEPTH = 4
WINDOWS = [3, 6, 12, 24, 48, 96, 168]
FEATURES = ["open", "high", "low", "close", "volume", "trades", "ret", "vwap", "taker_ratio"]
DERIVATIVE_FEATURES = ["funding_rate", "open_interest", "ls_account"]  # alpha.data.add_derivatives로 붙인 경우만

# 이름: (함수, 수식 인자 수, 윈도우 인자 여부)
OPS = {
    "neg": (lambda x: -x, 1, False),
    "abs": (lambda x: x.abs(), 1, False),
    "slog": (lambda x: np.sign(x) * np.log1p(x.abs()), 1, False),
    "add": (lambda a, b: a + b, 2, False),
    "sub": (lambda a, b: a - b, 2, False),
    "mul": (lambda a, b: a * b, 2, False),
    "div": (lambda a, b: a / b, 2, False),
    "ts_mean": (lambda x, d: x.rolling(d).mean(), 1, True),
    "ts_std": (lambda x, d: x.rolling(d).std(), 1, True),
    "ts_delta": (lambda x, d: x - x.shift(d), 1, True),
    "ts_rank": (lambda x, d: x.rolling(d).rank(pct=True), 1, True),
    "ts_zscore": (lambda x, d: (x - x.rolling(d).mean()) / x.rolling(d).std(), 1, True),
    "ts_max": (lambda x, d: x.rolling(d).max(), 1, True),
    "ts_min": (lambda x, d: x.rolling(d).min(), 1, True),
    "ts_corr": (lambda a, b, d: a.rolling(d).corr(b), 2, True),
}


def features(candles: pd.DataFrame) -> dict[str, pd.Series]:
    c = candles
    volume = c["volume"].where(c["volume"] > 0)
    return {
        "open": c["open"],
        "high": c["high"],
        "low": c["low"],
        "close": c["close"],
        "volume": c["volume"],
        "trades": c["trades"].astype("float64"),
        "ret": c["close"].pct_change(),
        "vwap": c["quote_volume"] / volume,
        "taker_ratio": c["taker_buy_volume"] / volume,
    } | {name: c[name] for name in DERIVATIVE_FEATURES if name in c}


def compute(e, feats: dict[str, pd.Series]) -> pd.Series:
    if isinstance(e, str):
        return feats[e]
    op, *args = e
    fn, n, _ = OPS[op]
    out = fn(*(compute(a, feats) for a in args[:n]), *args[n:])
    return out.replace([np.inf, -np.inf], np.nan)


def to_str(e) -> str:
    if isinstance(e, (str, int)):
        return str(e)
    op, *args = e
    return f"{op}({', '.join(to_str(a) for a in args)})"


def _children(e):
    return [] if isinstance(e, str) else e[1 : 1 + OPS[e[0]][1]]


def depth(e) -> int:
    return 0 if isinstance(e, str) else 1 + max(depth(c) for c in _children(e))


def size(e) -> int:
    return 1 + sum(size(c) for c in _children(e))


def _paths(e, path=()):
    yield path
    for i, c in enumerate(_children(e), start=1):
        yield from _paths(c, path + (i,))


def _get(e, path):
    for i in path:
        e = e[i]
    return e


def _replace(e, path, new):
    if not path:
        return new
    i = path[0]
    return e[:i] + (_replace(e[i], path[1:], new),) + e[i + 1 :]


def random_expr(rng: random.Random, max_depth: int):
    if max_depth == 0 or rng.random() < 0.3:
        return rng.choice(FEATURES)
    op = rng.choice(list(OPS))
    _, n, has_window = OPS[op]
    args = [random_expr(rng, max_depth - 1) for _ in range(n)]
    return (op, *args, rng.choice(WINDOWS)) if has_window else (op, *args)


def mutate(rng: random.Random, e):
    path = rng.choice(list(_paths(e)))
    node = _get(e, path)
    if not isinstance(node, str) and OPS[node[0]][2] and rng.random() < 0.5:
        new = node[:-1] + (rng.choice(WINDOWS),)  # 윈도우만 바꿔 미세 조정
    else:
        new = random_expr(rng, MAX_DEPTH - len(path))
    return _replace(e, path, new)


def crossover(rng: random.Random, a, b):
    path = rng.choice(list(_paths(a)))
    donors = [_get(b, p) for p in _paths(b) if depth(_get(b, p)) <= MAX_DEPTH - len(path)]
    return _replace(a, path, rng.choice(donors))
