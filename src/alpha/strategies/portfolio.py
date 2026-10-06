"""실거래 포트폴리오: 모멘텀 결합(momentum.LIVE_OPTIONS)과 펀딩 캐리(carry.LIVE_OPTIONS)를 한 계좌에서 비중으로 합친다.

두 전략이 같은 코인을 거래하므로 코인마다 순비중으로 상쇄되고(단방향 모드), 매일 실행할 때마다 평가금액 대비
두 전략 몫이 CARRY_SHARE 비율로 다시 맞춰진다. 결합 검증: docs/superpowers/specs/2026-10-06-funding-carry-research.md
"""
import pandas as pd

from alpha.strategies import carry, momentum

CARRY_SHARE = 0.5  # 평가금액 중 캐리 몫 (나머지는 모멘텀)


def live_weights(closes: pd.DataFrame, fundings: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """날짜 × 코인 비중: 전략별 몫(momentum, carry)과 합계(total)."""
    mom = (1 - CARRY_SHARE) * momentum.combined_weights(closes, **momentum.LIVE_OPTIONS)
    car = CARRY_SHARE * carry.carry_weights(closes, fundings, **carry.LIVE_OPTIONS)["W"]
    return {"momentum": mom, "carry": car, "total": mom + car}
