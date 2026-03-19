# BuyHigh-SellLow (BHSL)

Algorithmic trading framework for backtesting and paper trading.

## Requirements

- Python 3.12+
- pip
- (Optional) PostgreSQL for historical/backtest data
- (Optional) Redis for live paper market data mode

## Quick Start

```bash
# 1) Create virtual environment
python -m venv .venv

# 2) Activate it
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 3) Install package + dev tooling
pip install -e ".[dev]"

# 4) Setup git hooks
pre-commit install

# 5) Run checks
pre-commit run --all-files
pytest
```

## Install Profiles

Use extras depending on your workflow:

```bash
# Backtest optimization
pip install -e ".[optimization]"

# Paper trading runtime
pip install -e ".[paper]"

# Visualization
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Update values for your machine/account

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS/Linux
# cp .env.example .env
```

Core variables (minimum):

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

Paper trading related values are documented in `.env.example`.

## Development Commands

```bash
# Lint
ruff check .

# Format
ruff format .

# Type-check
mypy src/

# Tests
pytest
```

## CI

GitHub Actions workflows are in `.github/workflows/`:

- `ci.yml`: lint, format check, mypy, tests
- `backtest_regression.yml`: scheduled/manual regression job

## Project Structure

```text
src/
  data/
  database/
  engine/
  metrics/
  optimization/
  paper/
  strategy/
  utils/

tests/
  data/
  engine/
  optimization/
  paper/
  strategy/

config/
  schemas/
  strategy_params/
```

## Notes

- Commit directly to `main` is blocked by pre-commit hook.
- Keep secrets in `.env` only (never commit).
