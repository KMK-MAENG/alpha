"""gplearn으로 BTCUSDT 1h 알파 수식을 자동 탐색한다.

수식 생성/진화만 gplearn(SymbolicRegressor + 커스텀 fitness)이 맡고, 입력 피처, 포지션 변환, 수수료,
구간 분할, valid 선별, 보고는 alpha.mining.search와 동일하다 (두 엔진을 같은 조건에서 비교하기 위함).

실행 (프로젝트 루트에서): uv run python -m alpha.mining.gplearn_search [--pop-size N] [--generations N] [--seed N]
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn._program import _Program
from gplearn.fitness import make_fitness
from gplearn.functions import _function_map, make_function
from gplearn.genetic import SymbolicRegressor

from alpha.data import load_candles
from alpha.mining import expr, search


def ts_function(name: str, fn, arity: int, d: int):
    """시계열 연산자 fn(*series, d)를 윈도우 d로 고정한 gplearn 함수. 워밍업 NaN은 0으로 채운다."""

    def apply(*xs):
        out = fn(*(pd.Series(x) for x in xs), d).to_numpy()
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # gplearn은 함수 시그니처로 인자 수를 확인하므로 *args를 쓸 수 없다
    f = (lambda x: apply(x)) if arity == 1 else (lambda a, b: apply(a, b))
    return make_function(function=f, name=f"{name}_{d}", arity=arity, wrap=False)


FUNCTIONS = ["add", "sub", "mul", "div", "neg", "abs", "log"] + [
    ts_function(name, fn, arity, d)
    for name, (fn, arity, has_window) in expr.OPS.items()
    if has_window
    for d in expr.WINDOWS
]


def inputs(candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(expr.features(candles)).ffill().fillna(0)
    ret_next = candles["close"].pct_change().shift(-1).fillna(0)
    return X, ret_next


def _by_name(function_set) -> dict:
    return {f: _function_map[f] for f in function_set if isinstance(f, str)} | {
        f.name: f for f in function_set if not isinstance(f, str)
    }


def execute(e, X: pd.DataFrame, function_set) -> pd.Series:
    """함수 이름 튜플 수식(예: ("ts_rank_24", ("sub", "close", "open")))을 gplearn 함수로 계산한다."""
    functions = _by_name(function_set)

    def ev(node):
        if isinstance(node, float):
            return np.full(len(X), node)
        if isinstance(node, str):
            return X[node].to_numpy()
        op, *args = node
        return functions[op](*(ev(a) for a in args))

    return pd.Series(ev(e), index=X.index).replace([np.inf, -np.inf], np.nan)


def _to_program(e, functions: dict, features: list[str]) -> list:
    """튜플 수식 → gplearn 전위 표기 리스트 (함수 객체, 피처 인덱스, 상수)."""
    if isinstance(e, float):
        return [e]
    if isinstance(e, str):
        return [features.index(e)]
    op, *args = e
    return [functions[op]] + [node for a in args for node in _to_program(a, functions, features)]


def _inject_seeds(est: SymbolicRegressor, seeds: dict, X: np.ndarray, y: np.ndarray) -> None:
    """0세대 앞쪽을 시드 수식으로 교체한다. gplearn은 초기 집단 지정을 지원하지 않아
    비공개 속성(_programs, _function_set 등)을 쓰므로 gplearn 0.4.3 기준이다."""
    functions = _by_name(est._function_set)
    for i, e in enumerate(seeds.values()):
        p = _Program(
            function_set=est._function_set,
            arities=est._arities,
            init_depth=est.init_depth,
            init_method=est.init_method,
            n_features=X.shape[1],
            const_range=est.const_range,
            metric=est._metric,
            p_point_replace=est.p_point_replace,
            parsimony_coefficient=est.parsimony_coefficient,
            random_state=np.random.RandomState(i),
            feature_names=est.feature_names,
            program=_to_program(e, functions, est.feature_names),
        )
        p.raw_fitness_ = p.raw_fitness(X, y, np.ones(len(y)))
        p.fitness_ = p.fitness(est.parsimony_coefficient)
        est._programs[0][i] = p


def _net_sharpe(y, y_pred, sample_weight):
    """train 구간 수수료 차감 Sharpe (롱/숏 중 나은 쪽)."""
    signal = pd.Series(y_pred).replace([np.inf, -np.inf], np.nan)
    pos, ret_next = search.to_position(signal), pd.Series(y)
    best = max(search.sharpe(search.pnl(pos, ret_next)), search.sharpe(search.pnl(-pos, ret_next)))
    return best if np.isfinite(best) else 0.0


def evolve(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric,
    pop_size: int,
    generations: int,
    seed: int,
    function_set=FUNCTIONS,
    seeds: dict | None = None,
    const_range: tuple[float, float] | None = None,
) -> list:
    """metric(y, y_pred, w)을 최대화하도록 진화시키고, 보관된 프로그램을 train fitness 내림차순으로 돌려준다.
    seeds({이름: 튜플 수식})가 있으면 0세대에 넣고 이어서 진화시킨다."""
    X = X_train.to_numpy()
    est = SymbolicRegressor(
        population_size=pop_size,
        generations=1 if seeds else generations,
        function_set=function_set,
        metric=make_fitness(function=metric, greater_is_better=True, wrap=False),
        parsimony_coefficient=search.PARSIMONY,
        const_range=const_range,
        stopping_criteria=np.inf,  # 기본값 0.0이면 Sharpe가 0만 넘어도 1세대에서 멈춘다
        feature_names=list(X_train.columns),
        random_state=seed,
        verbose=1,
    )
    est.fit(X, y_train)
    if seeds:
        _inject_seeds(est, seeds, X, y_train)
        est.set_params(generations=generations, warm_start=True)
        est.fit(X, y_train)

    # gplearn은 지난 세대 중 자손을 남긴 프로그램만 보관한다 (나머지는 None)
    programs = {str(p): p for gen in est._programs for p in gen if p is not None}
    return sorted(programs.values(), key=lambda p: p.fitness_, reverse=True)


def run(
    candles: pd.DataFrame,
    pop_size: int = 1000,
    generations: int = 20,
    seed: int = 0,
    function_set=FUNCTIONS,
    seeds: dict | None = None,
) -> pd.DataFrame:
    X, ret_next = inputs(candles)
    programs = evolve(
        search.period(X, "train"), search.period(ret_next, "train").to_numpy(), _net_sharpe,
        pop_size, generations, seed, function_set, seeds,
    )
    candidates = {
        str(p): pd.Series(p.execute(X.to_numpy()), index=X.index).replace([np.inf, -np.inf], np.nan)
        for p in programs[: search.N_CANDIDATES]
    }
    return search.report(search.select(candidates, ret_next), ret_next)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run(load_candles("1h"), args.pop_size, args.generations, args.seed)
    print(result.round(2).to_string())
    out = Path("data") / f"gplearn_alphas_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}.csv"
    result.to_csv(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
