"""텔레그램 보고. .env의 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID를 쓴다 (없으면 보내지 않는다).

전송 실패는 경고만 출력하고 예외를 내지 않는다 — 이미 끝난 매매에 영향을 주지 않기 위함.

설정 확인 (프로젝트 루트에서):
  uv run python -m alpha.live.notify --chat-id   # 봇에게 메시지를 보낸 뒤 실행하면 채팅 ID를 알려준다
  uv run python -m alpha.live.notify --test      # 테스트 메시지 전송
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

import alpha.config  # noqa: F401 — .env 로드

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LENGTH = 4000  # 텔레그램 메시지 한도는 4096자
MODE_LABEL = {"live": "실주문", "dry-run": "DRY-RUN"}


def send(text: str, token: str | None = None, chat_id: str | None = None) -> bool:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("텔레그램 설정(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)이 없어 보고를 보내지 않음")
        return False
    try:
        for start in range(0, len(text), MAX_LENGTH):
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": text[start : start + MAX_LENGTH]}).encode()
            urllib.request.urlopen(API.format(token=token, method="sendMessage"), data=data, timeout=10)
    except Exception as e:  # 알림 실패가 매매 결과를 바꾸지 않도록
        print(f"텔레그램 전송 실패: {e!r}")
        return False
    return True


def _usdt(x: float, sign: bool = False) -> str:
    return f"{x:+,.2f}" if sign else f"{x:,.2f}"


def pnl_line(label: str, now: float, before: dict | None, transfer: float) -> str | None:
    """손익 = 평가금액 변화 - 순입금액. 수익률 분모는 기준 잔고 + 추가 입금 (출금은 빼지 않는다)."""
    if not before or "balances" not in before:
        return None
    base = before["balances"]["equity"]
    pnl = now - base - transfer
    since = pd.Timestamp(before["time"]).tz_convert("Asia/Seoul").strftime("%m-%d")
    note = f", 입출금 {_usdt(transfer, True)} 제외" if transfer else ""
    return f"{label} {_usdt(pnl, True)} ({pnl / (base + max(transfer, 0.0)):+.2%}, {since} 대비{note})"


def format_report(record: dict, previous: dict | None, first: dict | None, transfers: dict | None = None) -> str:
    """실행 기록 → 보고 문구. previous/first는 같은 모드의 직전·첫 실행 기록, transfers는 각각 그 이후의
    순입금액 {"previous": ..., "first": ...} (손익에서 뺀다)."""
    transfers = transfers or {}
    kst = pd.Timestamp(record["time"]).tz_convert("Asia/Seoul")
    b = record["balances"]
    lines = [f"📊 alpha 모멘텀+캐리 [{MODE_LABEL[record['mode']]}] {kst:%Y-%m-%d %H:%M} KST (일봉 {record['last_bar']})"]
    balance = f"💰 잔고 {_usdt(b['equity'])} USDT"
    if "wallet" in b:
        balance += f" (지갑 {_usdt(b['wallet'])} + 미실현 {_usdt(b['unrealized'], True)}, 사용 가능 {_usdt(b['available'])})"
    lines.append(balance)
    lines += [
        x for x in (
            pnl_line("   전일", b["equity"], previous, transfers.get("previous", 0.0)),
            pnl_line("   누적", b["equity"], first, transfers.get("first", 0.0)),
        ) if x
    ]

    changes = record["changes"]
    entries = [c for c in changes if c["kind"] == "진입"]
    exits = [c for c in changes if c["kind"] == "청산"]
    lines.append(f"🟢 진입 {len(entries)}")
    lines += [f"  {c['side']} {c['symbol']} {c['quantity']:g} ≈ {_usdt(c['notional'])} USDT" for c in entries]
    lines.append(f"🔴 청산 {len(exits)}")
    for c in exits:
        pnl = f" → 손익 {_usdt(c['pnl'], True)} USDT" if c.get("pnl") is not None else ""
        pct = f" ({c['pnl_pct']:+.1%})" if c.get("pnl_pct") is not None else ""
        lines.append(f"  {c['side']} {c['symbol']} {c['quantity']:g}{pnl}{pct}")
    others = {k: [c for c in changes if c["kind"] == k] for k in ("전환", "증액", "감액")}
    if any(others.values()):
        parts = [f"{k} {len(v)}" for k, v in others.items() if v]
        amount = sum(c["notional"] for v in others.values() for c in v)
        lines.append(f"↕️ 조정 {', '.join(parts)} (≈ {_usdt(amount)} USDT)")
        lines += [f"  전환 {c['side']} {c['symbol']}" for c in others["전환"]]

    h = record["holdings"]
    lines.append(f"📦 보유 {h['long'] + h['short']}개 (롱 {h['long']} / 숏 {h['short']}) | "
                 f"노출 롱 {h['long_exposure']:.1%} 숏 {h['short_exposure']:.1%}")
    if "strategy_exposure" in record:  # 상쇄 전 전략별 목표 (이전 기록에는 없음)
        e = record["strategy_exposure"]
        lines.append(f"🎯 목표 | 모멘텀 롱 {e['momentum']['long']:.1%} 숏 {e['momentum']['short']:.1%} | "
                     f"캐리 롱 {e['carry']['long']:.1%} 숏 {e['carry']['short']:.1%}")
    if record["skipped"]:
        lines.append(f"⚪ 최소 주문 미만 건너뜀 {len(record['skipped'])}: {', '.join(record['skipped'])}")
    if record["failures"]:
        lines.append(f"🚨 주문 실패 {record['failures']}")
        lines += [f"  {r['order']['symbol']}: {r['error']}" for r in record["results"] if "error" in r]
    if record["ignored_positions"]:
        lines.append(f"ℹ️ 전략 밖 포지션(건드리지 않음): {record['ignored_positions']}")
    return "\n".join(lines)


def format_failure(mode: str, error: Exception) -> str:
    return (f"🚨 alpha 모멘텀+캐리 실행 실패 [{MODE_LABEL[mode]}]\n{type(error).__name__}: {error}\n"
            "주문은 나가지 않았거나 일부만 나갔을 수 있다. 서버 로그(logs/live/)를 확인할 것")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-id", action="store_true", help="봇에게 온 메시지에서 채팅 ID 찾기")
    parser.add_argument("--test", action="store_true", help="테스트 메시지 전송")
    args = parser.parse_args()
    if args.chat_id:
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        updates = json.load(urllib.request.urlopen(API.format(token=token, method="getUpdates"), timeout=10))
        chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("username") for u in updates["result"] if "message" in u}
        print(chats or "봇에게 먼저 아무 메시지나 보낸 뒤 다시 실행")
    if args.test:
        print("전송 성공" if send("✅ alpha 텔레그램 연결 테스트") else "전송 실패")


if __name__ == "__main__":
    main()
