# 실거래 배포 인수인계 (2026-09-30)

> **집의 Claude에게**: 이 문서는 회사 PC에서 진행한 작업의 인수인계다. 아래 "남은 작업"을 순서대로 진행하도록 도와줘.
> 코드 변경보다 배포·계정 설정 위주이고, 비밀값(API 키, 봇 토큰)은 사용자가 직접 입력한다.
> 전략 설계와 검증 근거는 `docs/superpowers/specs/2026-09-29-momentum-live-trading-design.md`에 있다.

## 1. 지금까지 한 것 (요약)

- **전략**: 바이낸스 USDT-M 선물 56개 코인, 일봉 추세추종. 듀얼 모멘텀 + 시계열 모멘텀을 각각 포트폴리오 변동성 30%로 맞춘 뒤
  자본 50:50으로 결합. 진입 확인(20·60·120일 모두 플러스여야 진입) + 시계열 엔진 숏(롱의 거울상). 롱/숏 비율은 신호에 맡김.
  실거래 설정은 `src/alpha/strategies/momentum.py`의 `LIVE_OPTIONS`.
- **기대치** (백테스트, 2021년 이후·상장 폐지 포함 코호트 기준): Sharpe 약 1.1, 연 약 22%, 최대낙폭 약 -23%. 과거 성과는 보장이 아니다.
- **실행 방식**: 매일 UTC 00:05(한국 09:05) cron으로 `python -m alpha.live.runner --live` 1회. 방금 마감된 일봉으로 목표 비중을
  계산하고 현재 포지션과의 차이만 시장가 주문. 장중에는 아무것도 하지 않는다.
- **안전장치**: 데이터 최신성, 총 노출 1.5배 상한, 주문 1건 ≤ 평가금액 50%, 헤지 모드면 중단, 전략 밖 코인 포지션은 건드리지 않음.
- **텔레그램**: 매 실행 후 보고(잔고, 전일·누적 손익(입출금 제외), 진입, 청산(손익), 조정, 보유 현황), 실패하면 경고.
  조회 봇(상시 실행, 조회 전용): `/balance /positions /pnl /last /status /help`.
- **테스트**: `uv run pytest -q` → 141개 통과 (가짜 거래소 기준). **실계좌로는 아직 한 번도 실행하지 않았다.**
- **결정 사항**: dry-run 기간 없이 바로 실주문. `.env.example`은 사용자가 의도적으로 삭제함 (서버에서 `.env`를 직접 만든다).

## 2. 남은 작업 (순서대로)

### A. 회사 PC에서 (떠나기 전 또는 원격으로)

- [ ] 남은 변경 커밋·푸시: `git status`로 확인 후
      `git add -A .env.example deploy/ docs/ && git commit -m "chore: EC2 setup script (live cron) and handoff notes" && git push origin master`
      (원격은 SSH `git@github.com:KMK-MAENG/alpha.git`, 인증 완료)
- [ ] (선택) 데이터 수집 스크립트 백업: `/home/kmk/datas/crypto/update.py`는 저장소 밖에 있다. 연구를 계속할 거면 저장소 안
      (예: `scripts/update.py`)으로 옮기거나 따로 보관. 실거래 서버에는 필요 없다 (실행기가 시세를 API로 직접 받는다).

### B. 바이낸스 (웹/앱)

- [ ] **API 키**: 선물(Futures) 권한 켜기, **출금 권한 끄기**, IP 제한 = EC2의 Elastic IP (C에서 만든 뒤 등록).
      회사 PC에서는 `-2015 Invalid API-key, IP, or permissions` 오류가 났다 — 선물 권한·IP 제한·키 복사 오류 중 하나.
- [ ] **선물 지갑에 USDT 이체**: 현물 지갑 → USDT-M 선물 지갑 (시드 약 2,000 USDT). 전략은 선물 지갑 USDT만 인식한다.
- [ ] **포지션 모드 = 단방향(One-way)**: 헤지 모드면 실행기가 중단한다.
- [ ] **증거금 자산 모드 = 단일 자산(Single-Asset)**: 멀티에셋 모드는 쓰지 않는다 (다른 코인을 증거금으로 쓰면 급락 때 위험).
- [ ] 이 계좌는 **전략 전용**으로: 56개 전략 코인은 수동 거래하지 않는다 (실행기가 목표대로 되돌림).

### C. AWS EC2 (콘솔)

- [ ] 리전 **도쿄(ap-northeast-1)** — 미국 리전은 바이낸스가 차단한다
- [ ] Ubuntu 24.04, **t3.micro (x86)**, gp3 16~20GB **암호화**
- [ ] 보안 그룹: 인바운드 SSH(22)만, **내 IP에서만**
- [ ] **Elastic IP** 할당·연결 → 그 IP를 바이낸스 API 키 IP 허용 목록에 추가
- [ ] 비용 참고: 대략 월 10~15달러 (시드 2,000 USDT 대비 연 7~9%). 더 싸게는 Lightsail(고정 IP 포함 월 5달러 수준)

### D. 서버 설정 (`deploy/setup_ec2.sh`, 여러 번 실행해도 안전)

- [ ] 스크립트 올리기: `scp -i <키.pem> deploy/setup_ec2.sh ubuntu@<Elastic IP>:~`
- [ ] 접속 후 `bash ~/setup_ec2.sh` — 두 번 멈춘다:
  1. **배포 키**: 출력된 공개키를 GitHub 저장소 → Settings → Deploy keys → Add (write access 끔) → 다시 실행
  2. **`.env`**: 템플릿이 만들어지면 `nano ~/alpha/.env`로 채움 → 다시 실행
     ```
     BINANCE_API_KEY=...
     BINANCE_API_SECRET=...
     TELEGRAM_BOT_TOKEN=...
     TELEGRAM_CHAT_ID=...
     ```
     (`CRYPTO_DATA_DIR`은 연구용이라 서버에서는 비워도 됨. `~/alpha`는 스크립트가 clone하므로 `.env`는 clone 이후에 만든다)
- [ ] 스크립트가 끝까지 가면: 조회 봇(systemd `alpha-bot`) 실행, 바이낸스 연결 확인(주문 없음), 계좌 설정(교차 증거금·레버리지 3배),
      **실주문 cron 등록**, 텔레그램 테스트 메시지 수신
- [ ] 텔레그램에서 `/help`, `/balance`, `/status` 확인

### E. 첫 실주문

- [ ] 다음 한국 09:05에 자동 실행. 바로 시작하려면 서버에서 `cd ~/alpha && ~/.local/bin/uv run python -m alpha.live.runner --live`
- [ ] 텔레그램 보고 확인 (첫날은 진입이 수십 건), `/positions`로 보유 확인. 바이낸스 앱에서도 포지션 확인
- [ ] 문제가 있으면: 텔레그램 실패 경고, `~/alpha/logs/live/cron.log`, `~/alpha/logs/live/*.json`, `journalctl -u alpha-bot -f`

## 3. 운용 메모

- **매일 09:05쯤 텔레그램 보고가 없으면 이상** (서버 꺼짐·cron 미실행은 알림이 오지 않는다). 나중에 healthchecks.io 같은 외부 감시 추가 고려.
- **입출금**: 다음 09:05 실행 때 자동으로 새 잔고에 맞춰진다. 바로 맞추려면 수동으로 `runner --live` 1회. 큰 출금은 포지션을 먼저 줄이는 게 안전.
  보고의 손익은 입출금을 뺀 값이다.
- **코드 갱신**: 로컬에서 커밋·푸시 → 서버에서 `bash ~/setup_ec2.sh` 다시 실행 (pull, 의존성, 봇 재시작, cron 재등록).
- **비상 정지**: 전 포지션 정리 명령은 아직 없다. 바이낸스 앱에서 포지션을 닫고 **반드시 cron도 끈다**
  (`crontab -e`에서 `alpha.live.runner` 줄 주석 처리). cron을 두면 다음 09:05에 다시 포지션을 잡는다.
- **손으로 `runner`를 실행할 때는 `--live`가 있어야 주문이 나간다** (없으면 계산·기록만).
- 누적 손익은 서버의 `logs/live/` 기록 기준이다. 서버를 옮기면 이 폴더를 함께 옮겨야 이어진다.

## 4. 참고: 주요 파일

| 파일 | 역할 |
|---|---|
| `src/alpha/strategies/momentum.py` | 목표 비중 계산 (`combined_weights`, `LIVE_OPTIONS`, 코인 목록 `UNIVERSE`) |
| `src/alpha/exchange/client.py` | 바이낸스 통신 (일봉, 규칙, 잔고, 포지션, 입출금, 시장가 주문, 계좌 설정) |
| `src/alpha/execution/order_manager.py`, `risk.py` | 주문 계산(최소 주문·수량 단위), 주문 전 안전 점검 |
| `src/alpha/live/runner.py` | 하루 1회 실행기 (`--live`, `--setup`, `--equity` 미리보기) |
| `src/alpha/live/notify.py`, `bot.py` | 텔레그램 보고, 조회 봇 |
| `src/alpha/backtest/momentum.py` | 검증 백테스트 (`--cross-section`, `--dynamic`, `--trend-variants`, `--shorts` 등) |
| `deploy/setup_ec2.sh`, `deploy/alpha-bot.service` | 서버 설정 스크립트, 조회 봇 systemd 서비스 |

## 5. 나중에 해볼 만한 것 (급하지 않음)

- 비상용 텔레그램 제어 명령 (`/pause`, `/flatten`) — 토큰 유출 시 위험하므로 확인 절차와 함께
- 실행 누락 감시 (healthchecks.io를 cron에 한 줄)
- 연구용 데이터 스크립트 `update.py`를 저장소로 이전
