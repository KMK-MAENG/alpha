# Alpha 폴더 구조 스캐폴딩 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/superpowers/specs/2026-08-05-alpha-folder-structure-design.md`에 정의된 최상위 레이아웃과 `src/alpha` 내부 레이어드 구조를 실제 디렉터리/파일로 만든다.

**Architecture:** 리서치(`notebooks/`, `data/`)와 실거래(`src/alpha` 레이어드 패키지: `exchange`/`strategies`/`execution`/`backtest`/`live`)를 분리하고, 각 패키지는 빈 `__init__.py`만 갖는 자리표시 상태로 둔다. 유일하게 실동작하는 코드는 `config.py`의 `.env` 로딩 헬퍼이며 TDD로 작성한다.

**Tech Stack:** Python 3.14, `uv` (패키지 매니저), `python-dotenv`, `pytest` (신규 dev dependency)

## Global Constraints

- 기존 `src/alpha` src-layout을 그대로 유지한다 (변경하지 않음).
- `data/`, `logs/`, `.env`는 git에 커밋되지 않아야 한다.
- `notebooks/`는 하위 폴더를 강제하지 않고 평평하게 둔다.
- 이번 계획에서 **만들지 않는 것**: `Dockerfile`/`docker-compose.yml`, `strategies/base.py`(전략 인터페이스), `exchange/client.py`·`exchange/stream.py`·`execution/order_manager.py`·`execution/risk.py`·`backtest/engine.py`·`live/runner.py`의 실제 내용 (해당 파일들은 이번 계획에서 생성하지 않는다 — 패키지 디렉터리와 `__init__.py`만 생성).
- 모든 패키지 디렉터리에는 `__init__.py`가 존재해야 import 가능하다.

---

### Task 1: 최상위 디렉터리 스캐폴딩 (`data/`, `notebooks/`, `logs/`) + `.gitignore` + `.env.example`

**Files:**
- Create: `data/.gitkeep`
- Create: `notebooks/.gitkeep`
- Create: `logs/.gitkeep`
- Create: `.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `data/`, `notebooks/`, `logs/` 디렉터리가 저장소에 존재함 (다음 태스크들이 이 디렉터리 구조를 전제로 함)

- [ ] **Step 1: 디렉터리와 `.gitkeep` 생성**

```bash
mkdir -p data notebooks logs
touch data/.gitkeep notebooks/.gitkeep logs/.gitkeep
```

- [ ] **Step 2: `.gitignore`에 `data/`, `logs/`, `.env` 규칙 추가**

`.gitignore` 끝에 다음을 추가한다 (기존 내용은 그대로 둔다):

```gitignore

# Environment variables
.env

# Local data & logs (keep the directory, ignore contents)
data/*
!data/.gitkeep
logs/*
!logs/.gitkeep
```

- [ ] **Step 3: `.env.example` 생성**

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
```

- [ ] **Step 4: 결과 확인**

Run: `git status --short`
Expected: `notebooks/.gitkeep`가 추가 대상으로 보이고, `data/.gitkeep`·`logs/.gitkeep`는 `.gitignore` 규칙에 의해 `!data/.gitkeep`/`!logs/.gitkeep` 예외로 여전히 추가 대상으로 보임. `data/`나 `logs/` 하위에 다른 파일을 넣었다고 가정해도 무시되는지 확인하려면: `touch data/tmp.csv logs/tmp.log && git status --short data logs` → 두 파일 다 나타나지 않아야 함. 확인 후 `rm data/tmp.csv logs/tmp.log`로 정리.

- [ ] **Step 5: Commit**

```bash
git add data/.gitkeep notebooks/.gitkeep logs/.gitkeep .env.example .gitignore
git commit -m "chore: scaffold data/notebooks/logs directories and env template"
```

---

### Task 2: `src/alpha` 레이어드 서브패키지 스캐폴딩

**Files:**
- Create: `src/alpha/exchange/__init__.py`
- Create: `src/alpha/strategies/__init__.py`
- Create: `src/alpha/execution/__init__.py`
- Create: `src/alpha/backtest/__init__.py`
- Create: `src/alpha/live/__init__.py`

**Interfaces:**
- Produces: `alpha.exchange`, `alpha.strategies`, `alpha.execution`, `alpha.backtest`, `alpha.live` — 각각 import 가능한 빈 패키지. 이후 실제 코드(예: `alpha.exchange.client`)를 추가할 자리.

- [ ] **Step 1: 디렉터리와 빈 `__init__.py` 생성**

```bash
mkdir -p src/alpha/exchange src/alpha/strategies src/alpha/execution src/alpha/backtest src/alpha/live
touch src/alpha/exchange/__init__.py src/alpha/strategies/__init__.py src/alpha/execution/__init__.py src/alpha/backtest/__init__.py src/alpha/live/__init__.py
```

- [ ] **Step 2: import 가능한지 확인**

Run: `uv run python -c "import alpha.exchange, alpha.strategies, alpha.execution, alpha.backtest, alpha.live; print('ok')"`
Expected: `ok` 출력, 에러 없음.

- [ ] **Step 3: Commit**

```bash
git add src/alpha/exchange src/alpha/strategies src/alpha/execution src/alpha/backtest src/alpha/live
git commit -m "chore: scaffold alpha layered subpackages (exchange/strategies/execution/backtest/live)"
```

---

### Task 3: `config.py` — `.env` 로딩 헬퍼 (TDD)

**Files:**
- Create: `src/alpha/config.py`
- Test: `tests/test_config.py`
- Modify: `pyproject.toml` (pytest를 dev dependency로 추가)

**Interfaces:**
- Consumes: `python-dotenv`의 `load_dotenv` (이미 의존성에 있음)
- Produces: `alpha.config.get_env(key: str) -> str` — 환경변수를 읽어 반환하고, 없으면 `ValueError`를 던짐. 이후 `exchange`/`live` 등에서 `get_env("BINANCE_API_KEY")` 형태로 재사용될 함수.

- [ ] **Step 1: pytest를 dev dependency로 추가**

```bash
uv add --dev pytest
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_config.py`:

```python
import pytest

from alpha import config


def test_get_env_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("ALPHA_TEST_KEY", "test-value")
    assert config.get_env("ALPHA_TEST_KEY") == "test-value"


def test_get_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ALPHA_TEST_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPHA_TEST_MISSING_KEY"):
        config.get_env("ALPHA_TEST_MISSING_KEY")
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'alpha.config'`로 FAIL (아직 `config.py`가 없음)

- [ ] **Step 4: `config.py` 최소 구현**

`src/alpha/config.py`:

```python
import os

from dotenv import load_dotenv

load_dotenv()


def get_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2개 테스트 모두 PASS

- [ ] **Step 6: Commit**

```bash
git add src/alpha/config.py tests/test_config.py pyproject.toml uv.lock
git commit -m "feat: add alpha.config.get_env for .env-backed settings"
```

---

## Self-Review Notes

- **Spec coverage:** 최상위 레이아웃(`data/`, `notebooks/`, `logs/`, `.env`/`.env.example`) → Task 1. `src/alpha` 5개 서브패키지 → Task 2. `config.py` → Task 3. `strategies/base.py` 인터페이스, `Dockerfile`, `notebooks/` 하위 구조, `exchange`/`execution`/`backtest`/`live`의 실제 로직은 스펙에서 명시적으로 범위 밖이므로 이 플랜에도 포함하지 않음. `tests/`는 Task 3에서 `test_config.py`와 함께 자연스럽게 생성됨 (별도 빈 폴더 태스크 불필요).
- **Placeholder scan:** 모든 스텝에 실행 가능한 정확한 명령/코드 포함. TBD 없음.
- **Type consistency:** `config.get_env(key: str) -> str`이 Task 3 전체에서 동일하게 사용됨.
