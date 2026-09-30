"""텔레그램 조회 봇 (상시 실행, 조회 전용). .env의 TELEGRAM_CHAT_ID에서 온 명령에만 응답한다.

명령: /balance 잔고 · /positions 보유 포지션 · /pnl 손익 · /last 마지막 실행 보고 · /status 실행 상태 · /help
매매 실행기(runner, cron 하루 1회)와 별도 프로세스이며 주문은 내지 않는다. 시작할 때 꺼져 있던 동안 쌓인 명령은 건너뛴다.

실행 (프로젝트 루트에서): uv run python -m alpha.live.bot   — 서버에서는 deploy/alpha-bot.service(systemd)로 상시 실행
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from alpha.config import get_env
from alpha.exchange.client import BinanceFutures
from alpha.live import notify, runner
from alpha.strategies.momentum import UNIVERSE

HELP = (
    "📖 alpha 조회 봇 (조회 전용)\n"
    "/balance — 잔고 (평가금액·지갑·미실현·사용 가능)\n"
    "/positions — 보유 포지션 (진입가·현재가·손익·비중)\n"
    "/pnl — 마지막 실행 이후·누적 손익 (입출금 제외)\n"
    "/last — 마지막 실행 보고 다시 보기\n"
    "/status — 마지막 실행 결과와 다음 실행 시각"
)


def _kst(t: str | pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("Asia/Seoul").strftime("%m-%d %H:%M")


def balance(exchange) -> str:
    b = exchange.balances()
    return (f"💰 평가금액 {b['equity']:,.2f} USDT\n지갑 {b['wallet']:,.2f} | 미실현 {b['unrealized']:+,.2f} | "
            f"사용 가능 {b['available']:,.2f}")


def positions(exchange) -> str:
    details = exchange.position_details()
    if details.empty:
        return "📦 보유 포지션 없음"
    prices, equity = exchange.mark_prices(), exchange.balances()["equity"]
    notional = (details["qty"].abs() * prices.reindex(details.index)).rename("notional")
    rows = details.join(notional).sort_values("notional", ascending=False)
    long, short = rows[rows["qty"] > 0], rows[rows["qty"] < 0]
    lines = [f"📦 보유 {len(rows)}개 (롱 {len(long)} / 숏 {len(short)}) | 노출 롱 {long['notional'].sum() / equity:.1%} "
             f"숏 {short['notional'].sum() / equity:.1%} | 미실현 합계 {rows['unrealized_pnl'].sum():+,.2f}"]
    for sym, r in rows.iterrows():
        side = "롱" if r["qty"] > 0 else "숏"
        cost = abs(r["qty"]) * r["entry_price"]
        tag = " (전략 밖)" if sym not in UNIVERSE else ""
        lines.append(f"{side} {sym} {abs(r['qty']):g} | {r['entry_price']:g} → {prices.get(sym, float('nan')):g} | "
                     f"{r['unrealized_pnl']:+,.2f} ({r['unrealized_pnl'] / cost:+.1%}) | {r['notional'] / equity:.1%}{tag}")
    return "\n".join(lines)


def pnl(exchange, log_dir: Path, now: pd.Timestamp) -> str:
    earlier = runner._records(log_dir, "live", now.isoformat())
    if not earlier:
        return "📈 실주문 기록이 아직 없음 (손익은 실주문 실행 기록 기준)"
    last, first = earlier[-1], earlier[0]
    equity = exchange.balances()["equity"]
    since_last = exchange.net_transfers(pd.Timestamp(last["time"]), now)
    since_first = since_last + sum(r.get("transfers_since_previous", 0.0) for r in earlier[1:])
    return "\n".join(
        [f"📈 현재 평가금액 {equity:,.2f} USDT",
         notify.pnl_line("마지막 실행 이후", equity, last, since_last),
         notify.pnl_line("누적", equity, first, since_first)]
    )


def _latest(log_dir: Path, pattern: str) -> dict | None:
    files = sorted(log_dir.glob(pattern), key=lambda p: json.loads(p.read_text())["time"])
    return json.loads(files[-1].read_text()) if files else None


def last(log_dir: Path) -> str:
    records = [r for r in (_latest(log_dir, "*_live.json"), _latest(log_dir, "*_dry-run.json")) if r]
    if not records:
        return "실행 기록 없음"
    return runner.build_report(max(records, key=lambda r: r["time"]), log_dir)


def status(log_dir: Path, now: pd.Timestamp) -> str:
    lines = ["🩺 실행 상태"]
    for mode, label in (("live", "실주문"), ("dry-run", "DRY-RUN")):
        r = _latest(log_dir, f"*_{mode}.json")
        if r:
            lines.append(f"마지막 {label} 성공: {_kst(r['time'])} KST | 주문 {len(r.get('orders', []))} | 실패 {r.get('failures', 0)}")
    failed = _latest(log_dir, "*_failed.json")
    if failed:
        lines.append(f"마지막 실행 실패: {_kst(failed['time'])} KST [{notify.MODE_LABEL[failed['mode']]}] {failed['error']}")
    next_run = now.floor("D") + pd.Timedelta(minutes=5)
    if next_run <= now:
        next_run += pd.Timedelta(days=1)
    lines.append(f"다음 실행 예정: {_kst(next_run)} KST")
    return "\n".join(lines)


def handle(text: str, exchange, log_dir: Path | None, now: pd.Timestamp | None = None) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    command = text.strip().split()[0].split("@")[0].lower() if text.strip() else ""
    try:
        if command == "/balance":
            return balance(exchange)
        if command == "/positions":
            return positions(exchange)
        if command == "/pnl":
            return pnl(exchange, log_dir, now)
        if command == "/last":
            return last(log_dir)
        if command == "/status":
            return status(log_dir, now)
        return HELP
    except Exception as e:  # 조회 실패가 봇을 멈추지 않도록
        return f"⚠️ 오류: {type(e).__name__}: {e}"


def process_updates(updates: list[dict], chat_id: str, reply, answer) -> int | None:
    """설정된 채팅에서 온 메시지에만 답하고, 다음에 받을 update offset을 돌려준다."""
    offset = None
    for update in updates:
        offset = update["update_id"] + 1
        message = update.get("message")
        if message and str(message["chat"]["id"]) == str(chat_id) and message.get("text"):
            reply(answer(message["text"]))
    return offset


def _get_updates(token: str, offset: int | None, timeout: int) -> list[dict]:
    query = urllib.parse.urlencode({"timeout": timeout} | ({"offset": offset} if offset is not None else {}))
    url = notify.API.format(token=token, method="getUpdates") + "?" + query
    return json.load(urllib.request.urlopen(url, timeout=timeout + 10))["result"]


def main() -> None:
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        sys.exit(".env에 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID를 설정해야 한다")
    exchange = BinanceFutures(get_env("BINANCE_API_KEY"), get_env("BINANCE_API_SECRET"))
    backlog = _get_updates(token, offset=-1, timeout=0)  # 꺼져 있던 동안 쌓인 명령은 건너뛴다
    offset = backlog[-1]["update_id"] + 1 if backlog else None
    print("텔레그램 조회 봇 시작", flush=True)
    while True:
        try:
            updates = _get_updates(token, offset, timeout=50)
            offset = process_updates(
                updates, chat_id,
                reply=lambda text: notify.send(text, token, chat_id),
                answer=lambda text: handle(text, exchange, runner.LOG_DIR),
            ) or offset
        except Exception as e:  # 네트워크 오류 등 — 잠시 쉬고 다시 받는다
            print(f"업데이트 수신 실패: {e!r}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
