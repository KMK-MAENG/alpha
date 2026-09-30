"""유전 프로그래밍으로 BTCUSDT 1h 알파 수식을 자동 탐색한다.

1. train 구간의 수수료 차감 Sharpe(복잡도 패널티 포함)로 수식 집단을 진화시킨다.
2. 지금까지 평가한 수식 중 train 상위 후보를 valid 구간에서 거르고, 서로 상관이 낮은 것만 남긴다.
3. 선택된 알파의 test(홀드아웃) 성과를 보고한다. test는 탐색/선택에 쓰이지 않는다.

실행 (프로젝트 루트에서): uv run python -m alpha.mining.search [--pop-size N] [--generations N] [--seed N]
"""
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.data import load_candles
from alpha.mining import expr

FEE = 0.0005  # 포지션 1단위 변경당 테이커 수수료
BARS_PER_YEAR = 24 * 365
Z_WINDOW = 720  # 신호 → 포지션 정규화 윈도우 (30일)
PARSIMONY = 0.01  # 노드 1개당 fitness 패널티
N_CANDIDATES = 300  # valid 검증에 올릴 train 상위 후보 수
MIN_VALID_SHARPE = 0.5
MAX_CORR = 0.7
MAX_ALPHAS = 10
SPLITS = {
    "train": ("2019-09-01", "2023-06-30"),
    "valid": ("2023-07-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}


def set_interval(interval: str) -> None:
    """봉 크기(예: "4h")에 맞춰 연간 봉 수와 정규화 윈도우(30일)의 봉 수를 바꾼다. 기본값은 1h 기준."""
    global BARS_PER_YEAR, Z_WINDOW
    bar = pd.Timedelta(interval)
    BARS_PER_YEAR = int(pd.Timedelta(days=365) / bar)
    Z_WINDOW = int(pd.Timedelta(days=30) / bar)


def to_position(signal: pd.Series) -> pd.Series:
    z = (signal - signal.rolling(Z_WINDOW).mean()) / signal.rolling(Z_WINDOW).std()
    return (z / 2).clip(-1, 1).fillna(0)


def pnl(pos: pd.Series, ret_next: pd.Series) -> pd.Series:
    return pos * ret_next - FEE * pos.diff().abs().fillna(0)


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return returns.mean() / std * np.sqrt(BARS_PER_YEAR) if std > 0 else np.nan


def period(x: pd.Series, name: str) -> pd.Series:
    start, end = SPLITS[name]
    return x.loc[start:end]


def backtest(signal: pd.Series, ret_next: pd.Series):
    """train에서 더 나은 방향(롱/숏)으로 고정한 (방향, 포지션, 수익) 시계열."""
    pos = to_position(signal)
    if sharpe(period(pnl(-pos, ret_next), "train")) > sharpe(period(pnl(pos, ret_next), "train")):
        return -1, -pos, pnl(-pos, ret_next)
    return 1, pos, pnl(pos, ret_next)


def search(feats, ret_next, pop_size, generations, seed):
    """GP로 진화시키고, 평가한 모든 수식을 {문자열: (train fitness, 수식)}으로 돌려준다."""
    rng = random.Random(seed)
    evaluated = {}

    def fitness(e):
        key = expr.to_str(e)
        if key not in evaluated:
            s = sharpe(period(backtest(expr.compute(e, feats), ret_next)[2], "train"))
            evaluated[key] = (s - PARSIMONY * expr.size(e) if np.isfinite(s) else -np.inf, e)
        return evaluated[key][0]

    def tournament(pop):
        return max(rng.sample(pop, 5), key=fitness)

    pop = [expr.random_expr(rng, expr.MAX_DEPTH) for _ in range(pop_size)]
    for gen in range(generations):
        pop.sort(key=fitness, reverse=True)
        print(f"gen {gen:2d} | best {fitness(pop[0]):.2f} {expr.to_str(pop[0])} | 평가 {len(evaluated):,}개", flush=True)
        children = pop[: pop_size // 10]  # 상위 10%는 그대로 유지
        while len(children) < pop_size:
            a = tournament(pop)
            if rng.random() < 0.7:
                children.append(expr.crossover(rng, a, tournament(pop)))
            else:
                children.append(expr.mutate(rng, a))
        pop = children
    for e in pop:
        fitness(e)
    return evaluated


def select(candidates: dict[str, pd.Series], ret_next) -> dict[str, pd.Series]:
    """후보 {이름: 신호} 중 valid Sharpe 기준을 넘고, 서로 포지션 상관이 낮은 알파만 고른다."""
    passed = []
    for name, signal in candidates.items():
        _, pos, returns = backtest(signal, ret_next)
        valid = sharpe(period(returns, "valid"))
        if valid >= MIN_VALID_SHARPE:
            passed.append((valid, name, pos.loc[: SPLITS["valid"][1]]))
    passed.sort(key=lambda p: p[0], reverse=True)

    chosen = []
    for _, name, pos in passed:
        if all(abs(pos.corr(p)) < MAX_CORR for _, p in chosen):
            chosen.append((name, pos))
        if len(chosen) == MAX_ALPHAS:
            break
    return {name: candidates[name] for name, _ in chosen}


def _stats(pos, returns) -> dict:
    return {name: sharpe(period(returns, name)) for name in SPLITS} | {
        "turnover": pos.diff().abs().mean() * BARS_PER_YEAR
    }


def report(alphas: dict[str, pd.Series], ret_next) -> pd.DataFrame:
    """알파별, 동일가중 조합(combo), 단순 보유(buy_and_hold)의 구간별 Sharpe와 연간 회전율."""
    rows, positions = {}, []
    for name, signal in alphas.items():
        direction, pos, returns = backtest(signal, ret_next)
        positions.append(pos)
        rows[name] = {"dir": direction} | _stats(pos, returns)
    if positions:
        combo = pd.concat(positions, axis=1).mean(axis=1)
        rows["combo"] = _stats(combo, pnl(combo, ret_next))
    hold = pd.Series(1.0, index=ret_next.index)
    rows["buy_and_hold"] = _stats(hold, pnl(hold, ret_next))
    return pd.DataFrame.from_dict(rows, orient="index")


def run(candles: pd.DataFrame, pop_size: int = 300, generations: int = 30, seed: int = 0) -> pd.DataFrame:
    feats = expr.features(candles)
    ret_next = candles["close"].pct_change().shift(-1).fillna(0)
    evaluated = search(feats, ret_next, pop_size, generations, seed)
    top = sorted(evaluated.values(), key=lambda v: v[0], reverse=True)[:N_CANDIDATES]
    candidates = {expr.to_str(e): expr.compute(e, feats) for _, e in top}
    return report(select(candidates, ret_next), ret_next)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=300)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run(load_candles("1h"), args.pop_size, args.generations, args.seed)
    print(result.round(2).to_string())
    out = Path("data") / f"alphas_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}.csv"
    result.to_csv(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
