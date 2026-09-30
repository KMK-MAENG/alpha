"""여러 신호를 하나의 포지션으로 융합해 백테스트한다.

후보는 101 Alphas 변환 수식 99개다 (이 데이터에 맞춰 만든 수식이 아니라 train 성과로 골라도 편향이 작다).
1. train 수수료 전(gross) Sharpe가 MIN_TRAIN_GROSS 이상인 신호만 남기고, 방향(롱/숏)도 train에서 정한다.
2. 융합: 동일가중 평균(equal) 또는 train 선형회귀 가중치(ols).
3. 포지션을 EMA(반감기 H봉)로 스무딩해 회전율을 줄인다.
4. 융합 방식과 H는 valid 수수료 후 Sharpe로 고르고, test는 고른 설정 하나만 보고한다.

실행 (프로젝트 루트에서): uv run python -m alpha.mining.combine [--interval 4h]
"""
import argparse

import numpy as np
import pandas as pd

from alpha.data import load_candles
from alpha.mining import gplearn_search, search, wq101

MIN_TRAIN_GROSS = 0.5
METHODS = ["equal", "ols"]
HALFLIVES = [0, 6, 24, 72]  # 봉 단위, 0이면 스무딩 없음


def orient(signals: dict[str, pd.Series], ret_next: pd.Series) -> pd.DataFrame:
    """신호별 포지션을 train gross Sharpe가 양수인 방향으로 맞추고, 기준 미달 신호는 뺀다."""
    kept = {}
    for name, signal in signals.items():
        pos = search.to_position(signal)
        s = search.sharpe(search.period(pos * ret_next, "train"))
        if abs(s) >= MIN_TRAIN_GROSS:
            kept[name] = pos * np.sign(s)
    return pd.DataFrame(kept)


def fuse(positions: pd.DataFrame, ret_next: pd.Series, method: str) -> pd.Series:
    if method == "equal":
        return positions.mean(axis=1)
    train = search.period(positions, "train")
    weights, *_ = np.linalg.lstsq(train.to_numpy(), search.period(ret_next, "train").to_numpy(), rcond=None)
    return positions @ weights


def smooth(pos: pd.Series, halflife: int) -> pd.Series:
    return pos if halflife == 0 else pos.ewm(halflife=halflife).mean()


def _stats(pos: pd.Series, ret_next: pd.Series, splits) -> dict:
    net, gross = search.pnl(pos, ret_next), pos * ret_next
    return (
        {name: search.sharpe(search.period(net, name)) for name in splits}
        | {f"{name}_gross": search.sharpe(search.period(gross, name)) for name in splits}
        | {"turnover": pos.diff().abs().mean() * search.BARS_PER_YEAR}
    )


def run(signals: dict[str, pd.Series], ret_next: pd.Series):
    """(선택된 신호 이름, train/valid 그리드, 고른 설정, 고른 설정의 전 구간 성과)."""
    positions = orient(signals, ret_next)
    grid, table = {}, {}
    for method in METHODS:
        fused = search.to_position(fuse(positions, ret_next, method))
        for h in HALFLIVES:
            grid[(method, h)] = smooth(fused, h)
            table[(method, h)] = _stats(grid[(method, h)], ret_next, ["train", "valid"])
    table = pd.DataFrame.from_dict(table, orient="index").rename_axis(["method", "halflife"])
    best = table["valid"].idxmax()

    hold = pd.Series(1.0, index=ret_next.index)
    final = pd.DataFrame.from_dict(
        {f"{best[0]}, halflife={best[1]}": _stats(grid[best], ret_next, search.SPLITS),
         "buy_and_hold": _stats(hold, ret_next, search.SPLITS)},
        orient="index",
    )
    return list(positions.columns), table, best, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1h")
    args = parser.parse_args()

    search.set_interval(args.interval)
    X, ret_next = gplearn_search.inputs(load_candles(args.interval))
    signals = {f"wq#{n}": gplearn_search.execute(e, X, wq101.FUNCTIONS) for n, e in wq101.SEEDS.items()}
    selected, table, best, final = run(signals, ret_next)

    print(f"선택된 신호 {len(selected)}/{len(signals)}개: {', '.join(selected)}")
    print(f"\n[설정 비교: train/valid만]\n{table.round(2).to_string()}")
    print(f"\n[valid 기준 선택: {best}]\n{final.round(2).to_string()}")


if __name__ == "__main__":
    main()
