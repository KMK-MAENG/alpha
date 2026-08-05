# Alpha 프로젝트 폴더 구조 설계

## 배경

`alpha`는 바이낸스에서 algo trading을 하기 위한 프로젝트다. 두 가지 축이 공존해야 한다:

1. **리서치**: `ipynb`로 자유롭게 알파(전략 아이디어)를 탐색하고 검증
2. **실거래**: 검증된 전략을 상시 실행 프로세스로 돌림 (추후 VPS + Docker 배포 예정)

처음부터 여러 전략을 동시에(병렬로) 운영할 가능성을 염두에 두고 구조를 잡는다. 백테스트용 시세/거래 데이터는 로컬 파일(parquet/csv)로 저장한다.

기존 상태: `uv` 기반 Python 3.14 프로젝트, `src/alpha` src-layout 이미 존재, `pandas`/`numpy`/`matplotlib`/`ipykernel`/`python-dotenv` 의존성 설치됨. 트레이딩 관련 코드는 아직 없음.

## 접근 방식

전략들이 공통 인프라(거래소 연동, 주문 실행, 리스크 관리)를 공유하는 **레이어드 단일 패키지** 구조를 채택한다. 대안으로 전략별 독립 폴더(버티컬 슬라이스)도 검토했으나, 여러 전략이 같은 계좌/리스크 예산을 공유해야 한다는 요구사항상 주문 실행과 리스크 관리를 한 곳에서 통제하는 편이 낫다고 판단했다. 전략은 시세를 받아 매매 **신호**만 생성하고, 실제 주문 실행은 공통 `execution` 레이어가 전담한다.

## 최상위 레이아웃

```
alpha/
├── .env                  # API 키 등 (gitignore)
├── .env.example          # 필요한 환경변수 목록 (커밋됨)
├── data/                 # 로컬 시세/거래 데이터 (parquet/csv, gitignore)
├── notebooks/            # 자유 탐색용 ipynb (하위 구조 없이 평평하게, 파일 구성은 자유)
├── src/alpha/            # 검증된 코드 (기존 src-layout 유지)
├── tests/                # pytest
├── logs/                 # 실행 로그 (gitignore)
├── pyproject.toml / uv.lock
└── README.md / CLAUDE.md
```

## `src/alpha` 내부 구조

```
src/alpha/
├── __init__.py
├── config.py              # .env 로드, 공통 설정값
├── exchange/               # 바이낸스 API 연동 (REST + WebSocket) — 주문 실행은 하지 않음
│   ├── __init__.py
│   ├── client.py           # REST 클라이언트 wrapper
│   └── stream.py           # WebSocket 실시간 시세/체결 수신
├── strategies/              # 전략 모듈들 — 시세를 받아 신호만 생성, 직접 주문하지 않음
│   ├── __init__.py
│   └── base.py              # 공통 Strategy 인터페이스 (세부 설계는 구현 시점에 결정)
├── execution/               # 주문 실행 + 포지션/리스크 관리 (모든 전략 공통, 계좌 단위로 통제)
│   ├── __init__.py
│   ├── order_manager.py
│   └── risk.py
├── backtest/                 # 백테스트 엔진 — notebook과 live 양쪽에서 동일한 strategies/ 재사용
│   ├── __init__.py
│   └── engine.py
└── live/                      # 실거래 진입점 — 전략들을 로드해 상시 실행
    ├── __init__.py
    └── runner.py              # python -m alpha.live.runner
```

역할 구분:
- `exchange`: 바이낸스와의 통신만 담당
- `strategies`: 매수/매도 신호 생성만 담당
- `execution`: 신호를 받아 실제 주문 + 리스크 한도 체크 (전략 공유)
- `backtest`: 과거 데이터로 전략을 검증하는 엔진
- `live/runner.py`: 실행할 전략 목록을 설정에서 읽어 인스턴스화하고 상시 실행

## 전략 등록 규칙 (범위 밖)

전략을 `strategies/` 아래 어떻게 등록·로딩할지(예: `base.py` 인터페이스, 설정 기반 로딩)는 이번 설계에서 확정하지 않는다. `strategies/`, `execution/`, `backtest/`의 폴더 자리만 잡아두고, 실제 인터페이스 설계는 구현 시점에 별도로 다룬다.

## 데이터/노트북 연동

- notebook에서는 `src/alpha`를 일반 패키지처럼 import해서 사용한다 (uv 프로젝트이므로 `uv run jupyter` 또는 커널 등록 시 자동으로 잡힘).
- 검증된 로직은 `src/alpha/strategies/`로 옮기고, notebook은 "이 전략을 이렇게 검증했다"는 기록으로 남긴다.
- `data/`는 `.gitignore` 처리하여 대용량 시세 데이터를 git에 올리지 않는다.

## 테스트 & Docker

- `tests/`는 지금 빈 폴더(+ `__init__.py`)로 만들어두고, `execution/risk.py` 등 자금 관련 로직이 생기는 시점부터 채운다.
- Docker는 VPS 배포 시점에 필요하므로, `live/runner.py`가 실제로 동작하기 전까지는 만들지 않는다. 지금 만들면 내용 없는 빈 파일이 된다.

## 이번 설계에서 만들지 않는 것

- `Dockerfile` / `docker-compose.yml`
- `strategies/base.py`의 구체적인 인터페이스 (메서드 시그니처 등)
- `notebooks/` 하위 세부 폴더 구조
