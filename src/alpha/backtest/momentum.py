"""일봉 시계열 모멘텀 백테스트 (수수료·펀딩 반영).

- 날짜 t 종가(UTC 00시)에 포지션을 정하고 t+1일 동안 보유한다: 가격 손익 = pos_t × ret_{t+1}
- 수수료: 편도 FEE × |포지션 변화|. 펀딩: t+1일 정산(00·08·16시) 합 × pos_t (양의 펀딩이면 롱이 낸다).
  00시 정산은 리밸런싱과 같은 시각이라 새 포지션에 매겨 근사한다
- 비교: 계속 보유, 변동성 타게팅만 한 계속 보유(모멘텀 신호의 기여를 분리), 모멘텀 롱/숏·롱/무포지션,
  룩백별 단일 신호(파라미터 민감도), 연도별 성과. 연평균 수익·Sharpe는 일 수익률 평균 기준, MDD·연도별 수익은 복리

실행 (프로젝트 루트에서):
  uv run python -m alpha.backtest.momentum [--symbol BTCUSDT]   # 한 코인
  uv run python -m alpha.backtest.momentum --universe            # UNIVERSE 코인별 + 동일 비중 포트폴리오
  uv run python -m alpha.backtest.momentum --cross-section       # UNIVERSE 횡단면·듀얼 모멘텀
      [--interval 4h] [--lookbacks 20 60 120]                    # 봉 크기, 룩백(일)
  uv run python -m alpha.backtest.momentum --dynamic             # 고정 56개 vs 거래대금 동적 목록 (결합 50:50)
  uv run python -m alpha.backtest.momentum --indicators          # 시계열 모멘텀 신호를 지표(RSI, 이동평균 등)로 바꿔 비교
  uv run python -m alpha.backtest.momentum --xgboost             # 시계열 모멘텀 신호를 XGBoost 예측으로 바꿔 비교
  uv run python -m alpha.backtest.momentum --trend-variants      # 추세추종 보강(하방 변동성·진입 확인·순위 완충) 비교
  uv run python -m alpha.backtest.momentum --shorts              # 실거래 전략(롱 전용)에 숏을 더한 변형 비교
"""
import argparse

import numpy as np
import pandas as pd

from alpha.data import load_candles, load_funding, load_vision, vision_symbols
from alpha.strategies import indicators, ml, momentum

FEE = 0.0005  # 편도 테이커
SENSITIVITY_LOOKBACKS = [5, 10, 20, 40, 60, 90, 120, 180, 250]
UNIVERSE = momentum.UNIVERSE


def _bar(interval: str) -> pd.Timedelta:
    """바이낸스식 봉 표기(1d, 4h)를 Timedelta로 ('d' 단위 표기는 pandas에서 폐기 예정)."""
    return pd.Timedelta(interval.replace("d", "D"))


def backtest(close: pd.Series, funding: pd.Series, pos: pd.Series, interval: str = "1d") -> pd.DataFrame:
    held = pos.shift(1).fillna(0.0)
    bar_funding = funding.resample(_bar(interval)).sum()
    bar_funding.index = bar_funding.index.as_unit(close.index.unit)
    pnl = pd.DataFrame(
        {
            "price": held * close.pct_change().fillna(0.0),
            "funding": -held * bar_funding.reindex(close.index, fill_value=0.0),
            "fee": -FEE * held.diff().abs().fillna(held.abs()),
        }
    )
    pnl["total"] = pnl.sum(axis=1)
    return pnl


def summary(pnl: pd.DataFrame, pos: pd.Series | None = None, periods_per_year: int = 365) -> dict:
    total = pnl["total"]
    equity = (1 + total).cumprod()
    out = {
        "연평균 수익": total.mean() * periods_per_year,
        "연변동성": total.std() * np.sqrt(periods_per_year),
        "Sharpe": total.mean() / total.std() * np.sqrt(periods_per_year),
        "MDD": (equity / equity.cummax() - 1).min(),
        "펀딩": pnl["funding"].mean() * periods_per_year,
        "수수료": pnl["fee"].mean() * periods_per_year,
    }
    if pos is not None:
        out |= {"평균 |포지션|": pos.abs().mean(), "연 회전율": pos.diff().abs().mean() * 365}
    return out


def strategy_positions(close: pd.Series) -> dict[str, pd.Series]:
    vol = close.pct_change().rolling(momentum.VOL_WINDOW).std() * np.sqrt(365)
    return {
        "buy_and_hold": pd.Series(1.0, index=close.index),
        "vol_target_hold": (momentum.TARGET_VOL / vol).clip(upper=momentum.MAX_LEVERAGE),
        "momentum_long_short": momentum.position(close),
        "momentum_long_flat": momentum.position(close, long_only=True),
    }


def run_symbol(symbol: str, lookbacks: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """전략별 손익. 모든 전략을 기본 모멘텀 신호가 처음 나오는 날부터 같은 기간으로 잰다.
    lookbacks를 주면 모멘텀 두 전략만 그 룩백으로 바꾼다 (민감도용)."""
    close = load_candles("1d", symbol)["close"]
    funding = load_funding(symbol)
    positions = strategy_positions(close)
    start = positions["momentum_long_short"].first_valid_index()
    if lookbacks is not None:
        positions = {
            "momentum_long_short": momentum.position(close, lookbacks),
            "momentum_long_flat": momentum.position(close, lookbacks, long_only=True),
        }
    return {k: backtest(close.loc[start:], funding, pos.loc[start:]) for k, pos in positions.items()}


def _portfolio(pnls_by_symbol: dict[str, dict[str, pd.DataFrame]], strategy: str) -> pd.DataFrame:
    """그날 거래 가능한 코인 동일 비중 (구성 요소별 평균)."""
    frames = pd.concat({sym: pnls[strategy] for sym, pnls in pnls_by_symbol.items()}, axis=1)
    return frames.T.groupby(level=1).mean().T


def universe_main() -> None:
    pnls = {sym: run_symbol(sym) for sym in UNIVERSE}
    strategies = list(next(iter(pnls.values())))

    per_coin = pd.DataFrame({sym: {k: summary(v)["Sharpe"] for k, v in p.items()} for sym, p in pnls.items()}).T
    print(f"코인 {len(UNIVERSE)}개, 룩백 {momentum.LOOKBACKS}, 목표 변동성 {momentum.TARGET_VOL:.0%}, 편도 수수료 {FEE:.2%}")
    print(f"\n[코인별 Sharpe 분포]\n{per_coin.describe().loc[['25%', '50%', '75%', 'mean']].round(2).to_string()}")
    for name in ["momentum_long_flat", "momentum_long_short"]:
        beat = (per_coin[name] > per_coin["buy_and_hold"]).sum()
        beat_vt = (per_coin[name] > per_coin["vol_target_hold"]).sum()
        print(f"  {name}: 계속 보유보다 Sharpe 높은 코인 {beat}/{len(per_coin)}, 변동성 타게팅 보유보다 높은 코인 {beat_vt}/{len(per_coin)}")
    print(f"\n[코인별 Sharpe]\n{per_coin.round(2).sort_values('buy_and_hold').to_string()}")

    portfolio = {k: _portfolio(pnls, k) for k in strategies}
    table = pd.DataFrame.from_dict({k: summary(v) for k, v in portfolio.items()}, orient="index")
    print(f"\n[동일 비중 포트폴리오]\n{table.round(3).to_string()}")
    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in portfolio.items()})
    print(f"\n[포트폴리오 연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    rows = {}
    for n in SENSITIVITY_LOOKBACKS:
        by_symbol = {sym: run_symbol(sym, [n]) for sym in UNIVERSE}
        for name, key in [("롱/숏", "momentum_long_short"), ("롱/무포지션", "momentum_long_flat")]:
            rows[(n, name)] = summary(_portfolio(by_symbol, key))["Sharpe"]
    print(f"\n[포트폴리오 파라미터 민감도: 단일 룩백별 Sharpe]\n{pd.Series(rows).unstack().round(2).to_string()}")


def portfolio_pnl(
    closes: pd.DataFrame, fundings: dict[str, pd.Series], weights: pd.DataFrame, interval: str = "1d"
) -> pd.DataFrame:
    """코인별 비중으로 코인마다 backtest()를 돌려 구성 요소별로 합산한다 (리밸런싱 사이 비중 변동은 무시).
    가격 공백(상장 폐지 후 재상장 등)은 지우지 않는다 — 공백 전후 가격 차이가 하루 수익률로 잡히지 않고 0이 된다."""
    pnls = {}
    for sym in weights.columns[weights.abs().sum() > 0]:  # 한 번도 보유하지 않은 코인은 손익이 0
        close = closes[sym].loc[closes[sym].first_valid_index() :]
        pnls[sym] = backtest(close, fundings[sym], weights[sym].reindex(close.index).fillna(0.0), interval)
    frames = pd.concat(pnls, axis=1).fillna(0.0)
    return frames.T.groupby(level=1).sum().T


def cross_section_main(interval: str = "1d", lookbacks: list[float] = momentum.LOOKBACKS) -> None:
    bpd = int(pd.Timedelta(days=1) / _bar(interval))  # 하루 봉 개수
    ppy = 365 * bpd
    closes = pd.DataFrame({sym: load_candles(interval, sym)["close"] for sym in UNIVERSE})
    fundings = {sym: load_funding(sym) for sym in UNIVERSE}

    def run(weights: pd.DataFrame) -> pd.DataFrame:
        return portfolio_pnl(closes, fundings, weights, interval).loc[start:]

    def strategies(lbs: list[float]) -> dict[str, pd.DataFrame]:
        return {
            "xs_long_short": momentum.cross_sectional_weights(closes, lbs, bars_per_day=bpd),
            "xs_long_only": momentum.cross_sectional_weights(closes, lbs, long_only=True, bars_per_day=bpd),
            "dual": momentum.dual_momentum_weights(closes, lbs, bars_per_day=bpd),
            "basket": momentum.cross_sectional_weights(closes, lbs, quantile=1.0, long_only=True, bars_per_day=bpd),
            "ts_long_flat": momentum.time_series_weights(closes, lbs, bars_per_day=bpd),
        }

    weights = strategies(lookbacks)
    # 자격 코인이 MIN_COINS 이상이 된 봉부터 모든 전략을 같은 기간으로 잰다
    start = weights["basket"].abs().sum(axis=1).gt(0).idxmax()
    pnls = {k: run(w) for k, w in weights.items()}

    print(f"코인 {len(UNIVERSE)}개, {interval} 봉, {start.date()} ~ {closes.index[-1].date()} | 룩백 {lookbacks}일, "
          f"상·하위 {momentum.QUANTILE:.0%}, {momentum.REBALANCE_DAYS}일마다 리밸런싱, 편도 수수료 {FEE:.2%}")
    table = pd.DataFrame.from_dict({k: summary(v, periods_per_year=ppy) for k, v in pnls.items()}, orient="index")
    table["바스켓과 상관"] = [v["total"].corr(pnls["basket"]["total"]) for v in pnls.values()]
    table["평균 투자 비중"] = [w.loc[start:].sum(axis=1).mean() for w in weights.values()]
    print(table.round(3).to_string())
    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in pnls.items()})
    print(f"\n[연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    # 같은 위험 수준 비교: 롱 전략들에 포트폴리오 변동성 TARGET_VOL 타게팅을 씌운다
    long_strategies = ["dual", "xs_long_only", "basket", "ts_long_flat"]
    scaled = {k: momentum.vol_targeted(weights[k], closes, bars_per_day=bpd).loc[start:] for k in long_strategies}
    vt = {k: run(w) for k, w in scaled.items()}
    table = pd.DataFrame.from_dict({k: summary(v, periods_per_year=ppy) for k, v in vt.items()}, orient="index")
    table["평균 투자 비중"] = [w.sum(axis=1).mean() for w in scaled.values()]
    table["최대 투자 비중"] = [w.sum(axis=1).max() for w in scaled.values()]
    print(f"\n[포트폴리오 변동성 {momentum.TARGET_VOL:.0%} 타게팅 (배율 상한 {momentum.MAX_LEVERAGE:g}배)]\n{table.round(3).to_string()}")
    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in vt.items()})
    print(f"\n[변동성 타게팅 연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    # 성격이 반대인 듀얼(상승장)과 시계열(하락장 방어)을 결합: 자본 50:50, 그리고 결합 후 다시 변동성 타게팅
    def blends(dual_vt: pd.DataFrame, ts_vt: pd.DataFrame) -> dict[str, pd.DataFrame]:
        combo = (0.5 * dual_vt + 0.5 * ts_vt).reindex(closes.index, fill_value=0.0)
        return {
            "dual+vt": dual_vt,
            "ts_long_flat+vt": ts_vt,
            "combo_50_50": combo,
            "combo_50_50+vt": momentum.vol_targeted(combo, closes, bars_per_day=bpd),
        }

    blended = {k: w.loc[start:] for k, w in blends(scaled["dual"], scaled["ts_long_flat"]).items()}
    bt_blend = {k: run(w) for k, w in blended.items()}
    table = pd.DataFrame.from_dict({k: summary(v, periods_per_year=ppy) for k, v in bt_blend.items()}, orient="index")
    table["평균 투자 비중"] = [w.sum(axis=1).mean() for w in blended.values()]
    table["최대 투자 비중"] = [w.sum(axis=1).max() for w in blended.values()]
    corr = bt_blend["dual+vt"]["total"].corr(bt_blend["ts_long_flat+vt"]["total"])
    print(f"\n[듀얼 + 시계열 결합 (두 전략 일별 손익 상관 {corr:.2f})]\n{table.round(3).to_string()}")
    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in bt_blend.items()})
    print(f"\n[결합 연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    rows = {}
    for n in SENSITIVITY_LOOKBACKS:
        w = strategies([n])
        for name in ["xs_long_short", "xs_long_only", "dual"]:
            rows[(n, name)] = summary(run(w[name]), periods_per_year=ppy)["Sharpe"]
        dual_vt = momentum.vol_targeted(w["dual"], closes, bars_per_day=bpd)
        ts_vt = momentum.vol_targeted(w["ts_long_flat"], closes, bars_per_day=bpd)
        for name, blend_w in blends(dual_vt, ts_vt).items():
            rows[(n, name)] = summary(run(blend_w), periods_per_year=ppy)["Sharpe"]
    print(f"\n[파라미터 민감도: 단일 룩백(일)별 Sharpe, +vt는 변동성 타게팅]\n{pd.Series(rows).unstack().round(2).to_string()}")


def dynamic_main() -> None:
    """고정 56개 vs 매달 거래대금 상위 N개 (update.py --vision 데이터: 상장 폐지 코인 포함, 지난달까지)."""
    data = {sym: load_vision(sym) for sym in vision_symbols()}
    closes = pd.DataFrame({sym: c["close"] for sym, (c, _) in data.items()})
    volume = pd.DataFrame({sym: c["quote_volume"] for sym, (c, _) in data.items()})
    fundings = {sym: f for sym, (_, f) in data.items()}
    # 상장 폐지 뒤에도 가격 고정·거래량 0인 채움 기록이 이어지는 심볼이 많다 → 거래 없는 날은 기록 없음으로 본다
    closes = closes.where(volume > 0)
    last = closes.index[-1]
    delisted = [s for s in closes if closes[s].last_valid_index() < last - pd.Timedelta(days=7)]
    survivors = [s for s in closes if s not in delisted]

    def dynamic(n: int, symbols: list[str]) -> pd.DataFrame:
        u = momentum.liquidity_universe(closes[symbols], volume[symbols], top_n=n)
        return u.reindex(columns=closes.columns, fill_value=False)

    static = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    static[[s for s in UNIVERSE if s in closes]] = True
    # 고정 56개와 같은 "2020년 말 이전 상장" 코호트이되 이후 상장 폐지된 코인도 포함 — 2021년 시점에 알 수 있는 정보만
    first_bar = closes.apply(pd.Series.first_valid_index)
    cohort = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    cohort[list(first_bar.index[first_bar < pd.Timestamp("2021-01-01", tz="UTC")])] = True
    universes = {
        "static_56": static,
        "cohort_pre2021_with_delisted": cohort,
        "dynamic_top50": dynamic(50, list(closes.columns)),
        "dynamic_top50_survivors": dynamic(50, survivors),
        "dynamic_top30": dynamic(30, list(closes.columns)),
        "dynamic_top100": dynamic(100, list(closes.columns)),
        # 매달 상장(첫 기록) 후 3년 이상 된 거래 중인 코인 전부 — 순위 제한 없음
        "mature_3y": momentum.liquidity_universe(closes, volume, top_n=10**6, min_history_days=3 * 365),
    }
    weights = {k: momentum.combined_weights(closes, universe=u) for k, u in universes.items()}
    # 비교 대상 셋 모두 포지션이 생긴 날부터 같은 기간으로 잰다
    main = ["static_56", "dynamic_top50", "dynamic_top50_survivors"]
    start = max(weights[k].abs().sum(axis=1).gt(0).idxmax() for k in main)
    pnls = {k: portfolio_pnl(closes, fundings, w).loc[start:] for k, w in weights.items()}

    print(f"상장 폐지 포함 {closes.shape[1]}개 심볼 (그중 폐지 {len(delisted)}개), {start.date()} ~ {last.date()}, 결합 50:50")
    table = pd.DataFrame.from_dict({k: summary(v) for k, v in pnls.items()}, orient="index")
    table["평균 목록 코인"] = [universes[k].loc[start:].sum(axis=1).mean() for k in pnls]
    table["평균 보유 코인"] = [(weights[k].loc[start:] != 0).sum(axis=1).mean() for k in pnls]
    table["평균 투자 비중"] = [weights[k].loc[start:].sum(axis=1).mean() for k in pnls]
    print(table.round(3).to_string())

    dyn, fixed = universes["dynamic_top50"].loc[start:], universes["static_56"].loc[start:]
    overlap = (dyn & fixed).sum(axis=1) / dyn.sum(axis=1)
    print(f"\n동적 상위 50 중 고정 56개와 겹치는 비율: 평균 {overlap.mean():.0%} (처음 {overlap.iloc[0]:.0%} → 마지막 {overlap.iloc[-1]:.0%})")
    held = weights["dynamic_top50"].loc[start:]
    contrib = {
        s: (held[s].shift(1) * closes[s].pct_change()).loc[start:].sum()
        for s in delisted if held[s].abs().sum() > 0
    }
    if contrib:
        c = pd.Series(contrib).sort_values()
        print(f"상장 폐지 코인 {len(c)}개를 보유한 적 있음, 가격 손익 기여 합계 {c.sum():+.1%} "
              f"(가장 나쁜 3개: {', '.join(f'{k} {v:+.1%}' for k, v in c.head(3).items())})")

    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in pnls.items()})
    print(f"\n[연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    in_cohort = [c for c in cohort.columns[cohort.iloc[0]]]
    gone = [c for c in in_cohort if c in delisted]
    print(f"\n2020년 말 이전 상장 코호트 {len(in_cohort)}개 중 이후 상장 폐지 {len(gone)}개: {', '.join(gone)}")

    # 성숙 규칙은 2023년부터 계산 가능 — 같은 기간의 고정 56개와 비교
    mature_start = weights["mature_3y"].abs().sum(axis=1).gt(0).idxmax()
    rows = {k: summary(portfolio_pnl(closes, fundings, weights[k]).loc[mature_start:]) for k in ["static_56", "mature_3y"]}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["평균 목록 코인"] = [universes[k].loc[mature_start:].sum(axis=1).mean() for k in rows]
    print(f"\n[성숙 코인 규칙: {mature_start.date()} ~ {last.date()}]\n{table.round(3).to_string()}")


def indicators_main(symbol: str = "BTCUSDT") -> None:
    """시계열 모멘텀에서 신호만 바꿔 비교한다 (포지션 크기·비용은 같음). 개별 지표와 결합 신호를
    BTC 단일과 UNIVERSE 코인 전체(롱/무포지션)에서 재고, 기간을 반으로 나눠 안정성을 본다."""
    names = list(indicators.SIGNALS) + list(indicators.ENSEMBLES)
    close, funding = load_candles("1d", symbol)["close"], load_funding(symbol)
    variants = [(name, label, lo) for name in names for label, lo in [("롱/무포지션", True), ("롱/숏", False)]]
    positions = {(name, label): indicators.position(close, name, lo) for name, label, lo in variants}
    start = max(p.first_valid_index() for p in positions.values())  # 가장 느린 지표(200일선)가 준비된 날부터
    btc_pnl, rows = {}, {}
    for key, pos in positions.items():
        pos = pos.loc[start:]
        btc_pnl[key] = backtest(close.loc[start:], funding, pos)
        rows[key] = summary(btc_pnl[key], pos) | {"시장 참여율": (pos > 0).mean()}
    hold = pd.Series(1.0, index=close.loc[start:].index)
    rows[("buy_and_hold", "")] = summary(backtest(close.loc[start:], funding, hold), hold) | {"시장 참여율": 1.0}
    table = pd.DataFrame.from_dict(rows, orient="index")
    cols = ["Sharpe", "연평균 수익", "연변동성", "MDD", "평균 |포지션|", "연 회전율", "시장 참여율"]
    print(f"{symbol} 일봉 {start.date()} ~ {close.index[-1].date()} | 포지션 = 신호 × min(30% / 60일 변동성, 2배), 편도 수수료 {FEE:.2%}")
    print(table[cols].round(3).to_string())

    lf = {name: positions[(name, "롱/무포지션")].loc[start:] for name in indicators.SIGNALS}
    for ens, members in indicators.ENSEMBLES.items():
        corr = pd.DataFrame({m: lf[m] for m in members}).corr().to_numpy()
        pairs = corr[np.triu_indices(len(members), 1)]
        print(f"{ens}: 구성 신호 {len(members)}개의 BTC 포지션 평균 상관 {pairs.mean():.2f} (최소 {pairs.min():.2f})")

    # UNIVERSE 코인 전체에 같은 신호(롱/무포지션)를 적용 — BTC에서만 좋은 신호인지 확인
    closes = pd.DataFrame({sym: load_candles("1d", sym)["close"] for sym in UNIVERSE})
    fundings = {sym: load_funding(sym) for sym in UNIVERSE}
    all_pos = {name: closes.apply(lambda c, n=name: indicators.position(c, n, long_only=True)) for name in names}
    ready = pd.concat([p.notna() for p in all_pos.values()]).groupby(level=0).all()  # 코인별 모든 신호가 준비된 날
    port_start = ready.sum(axis=1).ge(momentum.MIN_COINS).idxmax()
    hold_sharpe = {}
    for sym in UNIVERSE:
        c = closes[sym].loc[ready[sym].idxmax() :].dropna()
        hold_sharpe[sym] = summary(backtest(c, fundings[sym], pd.Series(1.0, index=c.index)))["Sharpe"]
    port_pnl, rows = {}, {}
    for name, pos in all_pos.items():
        coin = {}
        for sym in UNIVERSE:
            c = closes[sym].loc[ready[sym].idxmax() :].dropna()
            coin[sym] = summary(backtest(c, fundings[sym], pos[sym].reindex(c.index)))["Sharpe"]
        coin = pd.Series(coin)
        held = pos.where(ready)
        w = held.fillna(0.0).div(held.notna().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        port_pnl[name] = portfolio_pnl(closes, fundings, w).loc[port_start:]
        rows[name] = {
            "코인별 Sharpe 중앙값": coin.median(),
            "계속 보유보다 나은 코인": f"{(coin > pd.Series(hold_sharpe)).sum()}/{len(coin)}",
            "동일 비중 포트폴리오 Sharpe": summary(port_pnl[name])["Sharpe"],
            "포트폴리오 연 수수료": summary(port_pnl[name])["수수료"],
        }
    rows["buy_and_hold"] = {"코인별 Sharpe 중앙값": pd.Series(hold_sharpe).median(), "계속 보유보다 나은 코인": "-"}
    print(f"\n[{len(UNIVERSE)}개 코인, 롱/무포지션, 포트폴리오는 {port_start.date()}부터]")
    print(pd.DataFrame.from_dict(rows, orient="index").round(3).to_string())

    halves = {"전반 (~2022)": (None, "2022-12-31"), "후반 (2023~)": ("2023-01-01", None)}
    candidates = ["return_sign_20_60_120", "rsi14_above_50", "bb_percent_b_20", *indicators.ENSEMBLES]
    rows = {}
    for name in candidates:
        row = {}
        for label, (a, b) in halves.items():
            row[f"BTC {label}"] = summary(btc_pnl[(name, "롱/무포지션")].loc[a:b])["Sharpe"]
            row[f"포트폴리오 {label}"] = summary(port_pnl[name].loc[a:b])["Sharpe"]
        rows[name] = row
    print(f"\n[기간별 Sharpe (롱/무포지션)]\n{pd.DataFrame.from_dict(rows, orient='index').round(2).to_string()}")


def xgboost_main() -> None:
    """UNIVERSE 패널로 워크포워드 학습한 XGBoost 예측의 부호를 시계열 모멘텀 신호로 쓰고 현재 신호와 비교한다."""
    candles = {sym: load_candles("1d", sym) for sym in UNIVERSE}
    fundings = {sym: load_funding(sym) for sym in UNIVERSE}
    data = ml.panel(candles, fundings)
    preds = ml.walk_forward_predictions(data)
    start = preds.index.get_level_values("date").min()
    xgb_signal = np.sign(preds).unstack("symbol")
    closes = pd.DataFrame({sym: c["close"] for sym, c in candles.items()})

    realized = data.loc[preds.index, "target"]
    both = pd.DataFrame({"pred": preds, "y": realized}).dropna()
    daily_ic = both.groupby(level="date").apply(lambda d: d["pred"].corr(d["y"], method="spearman"))
    btc = both.xs("BTCUSDT", level="symbol")
    print(f"워크포워드 예측 {start.date()} ~ {closes.index[-1].date()} (분기마다 재학습, 학습 행 {data['target'].notna().sum():,})")
    print(f"예측력: 코인 간 순위 상관(IC) 일평균 {daily_ic.mean():+.3f} (양수인 날 {(daily_ic > 0).mean():.0%}), "
          f"BTC 시계열 순위 상관 {btc['pred'].corr(btc['y'], method='spearman'):+.3f}")

    signals = {"xgboost": xgb_signal, "현재 신호": closes.apply(momentum.signal)}
    close, funding = closes["BTCUSDT"], fundings["BTCUSDT"]
    rows, btc_pnl = {}, {}
    for name, sig in signals.items():
        for label, lo in [("롱/무포지션", True), ("롱/숏", False)]:
            pos = indicators.size(close, sig["BTCUSDT"], lo).loc[start:]
            btc_pnl[(name, label)] = backtest(close.loc[start:], funding, pos)
            rows[(name, label)] = summary(btc_pnl[(name, label)], pos) | {"시장 참여율": (pos > 0).mean()}
    hold = pd.Series(1.0, index=close.loc[start:].index)
    rows[("buy_and_hold", "")] = summary(backtest(close.loc[start:], funding, hold), hold) | {"시장 참여율": 1.0}
    cols = ["Sharpe", "연평균 수익", "연변동성", "MDD", "평균 |포지션|", "연 회전율", "시장 참여율"]
    print(f"\n[BTCUSDT]\n{pd.DataFrame.from_dict(rows, orient='index')[cols].round(3).to_string()}")

    rows, port = {}, {}
    for name, sig in signals.items():
        pos = pd.DataFrame({sym: indicators.size(closes[sym], sig[sym], True) for sym in UNIVERSE}).loc[start:]
        w = pos.fillna(0.0).div(pos.notna().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        port[name] = portfolio_pnl(closes, fundings, w).loc[start:]
        rows[name] = summary(port[name])
    print(f"\n[{len(UNIVERSE)}개 코인 동일 비중 포트폴리오, 롱/무포지션]\n{pd.DataFrame.from_dict(rows, orient='index').round(3).to_string()}")

    yearly = pd.DataFrame(
        {f"BTC {name}": (1 + btc_pnl[(name, "롱/무포지션")]["total"]).groupby(lambda t: t.year).prod() - 1 for name in signals}
        | {f"포트폴리오 {name}": (1 + port[name]["total"]).groupby(lambda t: t.year).prod() - 1 for name in signals}
    )
    print(f"\n[연도별 수익 (복리, 롱/무포지션)]\n{yearly.round(3).to_string()}")

    # 피처 중요도 (설명용 — 라벨이 있는 전체 기간으로 한 번 더 학습한 모델, 성과 평가에는 쓰지 않음)
    features = [c for c in data.columns if c != "target"]
    labeled = data[data["target"].notna()]
    model = ml.XGBRegressor(**ml.PARAMS).fit(labeled[features], labeled["target"])
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print(f"\n[피처 중요도 (gain) 상위 10]\n{(gain / gain.sum()).head(10).round(3).to_string()}")


# 실행 전에 고정한 추세추종 보강 변형 (결합 50:50에 적용)
TREND_VARIANTS = {
    "V0 현재": {},
    "V1 하방 변동성": {"downside": True},
    "V2 진입 확인": {"confirm_entry": True},
    "V3 순위 완충": {"exit_quantile": 0.4},
    "V1+V2+V3": {"downside": True, "confirm_entry": True, "exit_quantile": 0.4},
}


def _holding_stats(weights: pd.DataFrame, closes: pd.DataFrame) -> dict:
    """코인별 연속 보유 구간(비중 > 0)의 길이와 손익 (포지션 × 다음 날 수익률)."""
    pnl = weights * closes.pct_change().shift(-1).reindex_like(weights)
    days, results = [], []
    for sym in weights:
        held = weights[sym] > 0
        group = (held != held.shift()).cumsum()[held]
        for _, idx in held[held].groupby(group).groups.items():
            days.append(len(idx))
            results.append(pnl[sym].loc[idx].sum())
    days, results = pd.Series(days), pd.Series(results)
    return {
        "보유 구간 중앙값(일)": days.median(),
        "보유 구간 평균(일)": days.mean(),
        "이긴 구간": (results > 0).mean(),
        "평균 이익/평균 손실": results[results > 0].mean() / -results[results < 0].mean(),
    }


def trend_variants_main() -> None:
    closes = pd.DataFrame({sym: load_candles("1d", sym)["close"] for sym in UNIVERSE})
    fundings = {sym: load_funding(sym) for sym in UNIVERSE}
    start = pd.Timestamp("2020-06-14", tz="UTC")
    rows, pnls = {}, {}
    for name, options in TREND_VARIANTS.items():
        w = momentum.combined_weights(closes, **options).loc[start:]
        pnls[name] = portfolio_pnl(closes, fundings, w).loc[start:]
        rows[name] = summary(pnls[name]) | {"연 회전율": w.diff().abs().sum(axis=1).mean() * 365} | _holding_stats(w, closes)
    print(f"[고정 56개, 결합 50:50, {start.date()} ~ {closes.index[-1].date()}]")
    print(pd.DataFrame.from_dict(rows, orient="index").round(3).to_string())
    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in pnls.items()})
    print(f"\n[연도별 수익 (복리)]\n{yearly.round(3).to_string()}")

    # 2020년 말 이전 상장 코호트 (이후 상장 폐지 포함, 거래량 0인 채움 기록 제외) — 미래 정보 없는 검증
    data = {sym: load_vision(sym) for sym in vision_symbols()}
    vclose = pd.DataFrame({s: c["close"].where(c["volume"] > 0) for s, (c, _) in data.items()})
    cohort = [s for s in vclose if vclose[s].first_valid_index() < pd.Timestamp("2021-01-01", tz="UTC")]
    vclose, vfund = vclose[cohort], {s: data[s][1] for s in cohort}
    rows = {}
    for name, options in TREND_VARIANTS.items():
        w = momentum.combined_weights(vclose, **options).loc["2021-01-01":]
        rows[name] = summary(portfolio_pnl(vclose, vfund, w).loc["2021-01-01":])
    print(f"\n[2020년 말 이전 상장 코호트 {len(cohort)}개 (폐지 포함), 2021-01-01 ~ {vclose.index[-1].date()}]")
    print(pd.DataFrame.from_dict(rows, orient="index").round(3).to_string())


def _short_variants(closes: pd.DataFrame, universe: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """실행 전에 고정한 숏 변형 (모두 진입 확인, 숏도 대칭으로 확인). S1이 실거래 설정(LIVE_OPTIONS)으로 채택됐다."""
    base = dict(universe=universe, confirm_entry=True)
    return {
        "S0 롱 전용": momentum.combined_weights(closes, **base),
        "S1 시계열 숏 (실거래)": momentum.combined_weights(closes, ts_short=True, **base),
        "S2 시계열+듀얼 숏": momentum.combined_weights(closes, allow_short=True, **base),
        "S3 S2+시장 필터": momentum.combined_weights(closes, allow_short=True, short_market="BTCUSDT", **base),
    }


def _short_report(title: str, closes: pd.DataFrame, fundings: dict, variants: dict, start: str, end: str | None = None) -> None:
    rows, yearly = {}, {}
    returns = closes.pct_change()
    for name, w in variants.items():
        w = w.loc[start:end]
        pnl = portfolio_pnl(closes, fundings, w).loc[start:end]
        held = w.shift(1)
        price = held * returns.reindex_like(held)
        years = len(pnl) / 365
        rows[name] = summary(pnl) | {
            "롱 노출": w.clip(lower=0).sum(axis=1).mean(),
            "숏 노출": -w.clip(upper=0).sum(axis=1).mean(),
            "롱 가격손익(연)": price.where(held > 0).sum().sum() / years,
            "숏 가격손익(연)": price.where(held < 0).sum().sum() / years,
        }
        yearly[name] = (1 + pnl["total"]).groupby(pnl.index.year).prod() - 1
    print(f"\n[{title}]\n{pd.DataFrame.from_dict(rows, orient='index').round(3).to_string()}")
    print(f"연도별 수익 (복리)\n{pd.DataFrame(yearly).round(3).to_string()}")


def shorts_main() -> None:
    closes = pd.DataFrame({sym: load_candles("1d", sym)["close"] for sym in UNIVERSE})
    fundings = {sym: load_funding(sym) for sym in UNIVERSE}
    _short_report("고정 56개 (지금 거래 중인 코인만 — 폭락해 사라진 코인이 없어 숏에 불리)",
                  closes, fundings, _short_variants(closes), "2020-06-14")

    data = {sym: load_vision(sym) for sym in vision_symbols()}
    vclose = pd.DataFrame({s: c["close"].where(c["volume"] > 0) for s, (c, _) in data.items()})
    volume = pd.DataFrame({s: c["quote_volume"] for s, (c, _) in data.items()})
    vfund = {s: f for s, (_, f) in data.items()}
    cohort = [s for s in vclose if vclose[s].first_valid_index() < pd.Timestamp("2021-01-01", tz="UTC")]
    _short_report(f"2020년 말 이전 상장 코호트 {len(cohort)}개 (상장 폐지 포함)",
                  vclose[cohort], vfund, _short_variants(vclose[cohort]), "2021-01-01")
    dynamic = momentum.liquidity_universe(vclose, volume, top_n=50)
    _short_report("매달 거래대금 상위 50 (상장 폐지·신규 코인 포함)",
                  vclose, vfund, _short_variants(vclose, universe=dynamic), "2021-01-01")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--universe", action="store_true", help="UNIVERSE 코인별 + 동일 비중 포트폴리오")
    parser.add_argument("--cross-section", action="store_true", help="UNIVERSE 횡단면·듀얼 모멘텀")
    parser.add_argument("--dynamic", action="store_true", help="고정 56개 vs 거래대금 동적 목록")
    parser.add_argument("--indicators", action="store_true", help="시계열 모멘텀 신호를 지표 기반으로 바꿔 비교")
    parser.add_argument("--xgboost", action="store_true", help="시계열 모멘텀 신호를 XGBoost 예측으로 바꿔 비교")
    parser.add_argument("--trend-variants", action="store_true", help="추세추종 보강 변형 비교")
    parser.add_argument("--shorts", action="store_true", help="실거래 전략에 숏을 더한 변형 비교")
    parser.add_argument("--interval", default="1d", help="--cross-section의 봉 크기 (1d, 4h)")
    parser.add_argument("--lookbacks", type=float, nargs="+", default=momentum.LOOKBACKS, help="--cross-section의 룩백(일)")
    args = parser.parse_args()
    if args.universe:
        universe_main()
        return
    if args.shorts:
        shorts_main()
        return
    if args.trend_variants:
        trend_variants_main()
        return
    if args.xgboost:
        xgboost_main()
        return
    if args.indicators:
        indicators_main(args.symbol)
        return
    if args.dynamic:
        dynamic_main()
        return
    if args.cross_section:
        cross_section_main(args.interval, args.lookbacks)
        return

    close = load_candles("1d", args.symbol)["close"]
    funding = load_funding(args.symbol)
    positions = strategy_positions(close)
    # 포지션은 전체 가격으로 계산하고, 모든 전략을 모멘텀 신호가 처음 나오는 날부터 같은 기간으로 비교한다
    # (룩백이 더 긴 민감도 신호는 신호가 생기기 전까지 무포지션)
    start = positions["momentum_long_short"].first_valid_index()
    period_close = close.loc[start:]
    positions = {k: v.loc[start:] for k, v in positions.items()}
    pnls = {k: backtest(period_close, funding, pos) for k, pos in positions.items()}

    print(f"{args.symbol} 일봉 {start.date()} ~ {period_close.index[-1].date()} | 룩백 {momentum.LOOKBACKS}, "
          f"목표 변동성 {momentum.TARGET_VOL:.0%}, 편도 수수료 {FEE:.2%}")
    table = pd.DataFrame.from_dict({k: summary(pnls[k], positions[k]) for k in positions}, orient="index")
    print(table.round(3).to_string())

    rows = {}
    for n in SENSITIVITY_LOOKBACKS:
        for name, long_only in [("롱/숏", False), ("롱/무포지션", True)]:
            pos = momentum.position(close, [n], long_only).loc[start:]
            rows[(n, name)] = summary(backtest(period_close, funding, pos), pos)["Sharpe"]
    sensitivity = pd.Series(rows).unstack()
    print(f"\n[파라미터 민감도: 단일 룩백별 Sharpe (같은 변동성 타게팅)]\n{sensitivity.round(2).to_string()}")

    yearly = pd.DataFrame({k: (1 + v["total"]).groupby(v.index.year).prod() - 1 for k, v in pnls.items()})
    print(f"\n[연도별 수익 (복리)]\n{yearly.round(3).to_string()}")


if __name__ == "__main__":
    main()
