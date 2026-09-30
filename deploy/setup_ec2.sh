#!/usr/bin/env bash
# alpha 실거래 서버 초기 설정 (EC2 Ubuntu 24.04, x86_64). 여러 번 실행해도 안전하다 — 준비가 필요한 단계에서 멈추면
# 안내대로 준비한 뒤 다시 실행하면 이어서 진행한다.
#
# 매매 cron은 실주문(--live)으로 등록한다. 등록 전에 API 키·단방향 모드를 확인하고 계좌를 설정한다
# (교차 증거금·레버리지 3배 — 실제 노출은 전략 비중이 정한다).
#
# 사용법 (서버에서, sudo 가능한 일반 사용자로):
#   scp deploy/setup_ec2.sh ubuntu@<서버 IP>:~      # 내 PC에서 스크립트만 먼저 올린다
#   bash ~/setup_ec2.sh                             # 처음 설정, 코드 갱신 모두 같은 명령
set -euo pipefail

REPO="git@github.com:KMK-MAENG/alpha.git"
APP_DIR="$HOME/alpha"
UV="$HOME/.local/bin/uv"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
stop() { printf '\n\033[1;33m[멈춤] %s\033[0m\n' "$*"; exit 1; }

step "1. 기본 패키지, 시간대(UTC)"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl ca-certificates cron >/dev/null
sudo timedatectl set-timezone UTC
timedatectl show -p NTPSynchronized --value | grep -q yes || echo "주의: 시간 동기화가 아직 안 됨 (바이낸스는 요청 시각이 어긋나면 거부한다)"

step "2. uv 설치"
[[ -x "$UV" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
"$UV" --version

step "3. GitHub 배포 키 (이 서버 전용, 읽기 전용)"
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[[ -f ~/.ssh/id_ed25519 ]] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "alpha-ec2"
grep -q "^github.com" ~/.ssh/known_hosts 2>/dev/null || ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
# ssh -T는 인증에 성공해도 종료 코드가 1이라, 파이프 대신 출력을 받아 검사한다 (pipefail이면 항상 실패로 보임)
auth=$(ssh -T git@github.com 2>&1 || true)
if ! grep -q "successfully authenticated" <<< "$auth"; then
    echo "아래 공개키를 GitHub 저장소 → Settings → Deploy keys → Add deploy key 에 등록 (Allow write access는 끈다):"
    echo
    cat ~/.ssh/id_ed25519.pub
    stop "등록한 뒤 이 스크립트를 다시 실행"
fi

step "4. 저장소 받기, 의존성 설치"
if [[ -d "$APP_DIR/.git" ]]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
"$UV" sync --no-dev
mkdir -p logs/live

step "5. .env (API 키·텔레그램 설정)"
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
    else
        printf 'BINANCE_API_KEY=\nBINANCE_API_SECRET=\nCRYPTO_DATA_DIR=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_CHAT_ID=\n' > .env
    fi
    chmod 600 .env
    stop "$APP_DIR/.env 를 채운 뒤 다시 실행 (BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
       CRYPTO_DATA_DIR는 연구용이라 비워도 된다. 기존 PC의 누적 손익을 이으려면 logs/live/ 도 복사해 둔다:
       scp -r ~/projects/alpha/logs/live ubuntu@<서버 IP>:~/alpha/logs/"
fi
chmod 600 .env
env_value() { grep -E "^$1=" .env | cut -d= -f2- | tr -d '"'"'"' '; }
for key in BINANCE_API_KEY BINANCE_API_SECRET TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
    [[ -n "$(env_value "$key")" ]] || stop ".env의 $key 가 비어 있음 — 채운 뒤 다시 실행"
done

step "6. 텔레그램 조회 봇 (systemd, 상시 실행)"
sed -e "s|^User=.*|User=$USER|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$APP_DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$UV run python -m alpha.live.bot|" \
    deploy/alpha-bot.service | sudo tee /etc/systemd/system/alpha-bot.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable alpha-bot >/dev/null 2>&1
sudo systemctl restart alpha-bot
sleep 3
systemctl is-active --quiet alpha-bot && echo "alpha-bot 실행 중" || stop "alpha-bot 시작 실패 — journalctl -u alpha-bot -n 50 확인"

step "7. 바이낸스 연결 확인 (주문 없음: 잔고·단방향 모드)"
PUBLIC_IP=$(curl -s https://checkip.amazonaws.com)
echo "이 서버의 공인 IP: $PUBLIC_IP  ← 바이낸스 API 키의 IP 허용 목록에 있어야 한다 (Elastic IP를 붙여 고정할 것)"
if ! "$UV" run python -c "
from alpha.config import get_env
from alpha.exchange.client import BinanceFutures
exchange = BinanceFutures(get_env('BINANCE_API_KEY'), get_env('BINANCE_API_SECRET'))
print('선물 지갑:', exchange.balances())
assert exchange.is_one_way(), '헤지 모드 계좌 — 바이낸스 선물 설정에서 단방향(One-way) 모드로 바꿀 것'
"; then
    stop "바이낸스 연결 실패 — -2015면 API 키의 선물 권한·IP 허용 목록($PUBLIC_IP)을 확인"
fi

step "8. 계좌 설정 (교차 증거금·레버리지 3배, 이미 설정돼 있으면 그대로)"
"$UV" run python -m alpha.live.runner --setup

step "9. 매매 cron (실주문, 매일 UTC 00:05 = 한국 09:05)"
JOB="5 0 * * * cd $APP_DIR && $UV run python -m alpha.live.runner --live >> logs/live/cron.log 2>&1"
( crontab -l 2>/dev/null | grep -v "alpha.live.runner" || true; echo "$JOB" ) | crontab -
crontab -l | grep "alpha.live.runner"

step "10. 텔레그램 확인"
"$UV" run python -m alpha.live.notify --test

step "완료"
cat <<EOF
- 첫 실주문: 다음 한국 시간 09:05 (결과는 텔레그램 보고). 바로 시작하려면:
    cd $APP_DIR && $UV run python -m alpha.live.runner --live
- 조회 봇: 텔레그램에서 /help        (상태: systemctl status alpha-bot / 로그: journalctl -u alpha-bot -f)
- 매매 로그: $APP_DIR/logs/live/cron.log, 실행 기록: $APP_DIR/logs/live/*.json
- 코드 갱신: 이 스크립트를 다시 실행 (git pull + 의존성 + 봇 재시작 + cron 재등록)
EOF
