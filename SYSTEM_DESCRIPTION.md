# BuyHigh-SellLow Trading System

> A professional algorithmic trading framework for backtesting, optimization, and paper trading on Vietnamese futures markets

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Management](#data-management)
  - [Backtesting](#backtesting)
  - [Optimization](#optimization)
  - [Walk-Forward Analysis](#walk-forward-analysis)
  - [Paper Trading](#paper-trading)
- [Strategy Development](#strategy-development)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

BuyHigh-SellLow is a production-grade algorithmic trading system designed for the Vietnamese VN30 Index Futures market. It provides a complete workflow from strategy development to live paper trading, with emphasis on robustness, testing, and risk management.

### Key Capabilities

- **Backtesting Engine**: Event-driven backtester with realistic order execution simulation
- **Strategy Framework**: Pluggable strategy architecture with built-in ORB (Opening Range Breakout) implementation
- **Optimization**: Optuna-based hyperparameter optimization with custom scoring functions
- **Walk-Forward Analysis**: Robust validation across expanding/shifting time windows
- **Paper Trading**: Production-ready paper trading with Redis market data and FIX protocol support
- **Risk Management**: Position sizing, stop-loss, take-profit, and daily loss limits
- **Data Pipeline**: Efficient tick-to-bar aggregation with caching and quality validation

---

## Features

### Backtesting
- ✅ Event-driven architecture for realistic simulation
- ✅ Multiple order types (Market, Limit) with TTL support
- ✅ Slippage and commission modeling
- ✅ Session-aware trading (morning/afternoon sessions)
- ✅ End-of-day position flattening
- ✅ Comprehensive performance metrics (Sharpe, Sortino, Max DD, etc.)
- ✅ Rich visualization (equity curves, drawdowns, trade analysis)

### Optimization
- ✅ Optuna TPE (Tree-structured Parzen Estimator) optimizer
- ✅ Custom composite scoring function (Sharpe - Drawdown - Trade Frequency)
- ✅ Parallel trial execution
- ✅ Automatic parameter space definition
- ✅ Best parameter persistence

### Paper Trading
- ✅ Three operating modes: LIVE, DRY-RUN, SIM
- ✅ Redis integration for real-time market data
- ✅ FIX protocol support for order execution
- ✅ Bar aggregation from tick data with quality validation
- ✅ Reconciliation and position tracking
- ✅ Session statistics and performance reporting
- ✅ Comprehensive test coverage (396 tests, 100% pass rate)

### Data Management
- ✅ PostgreSQL integration for historical tick data
- ✅ Efficient chunked data loading (30-day chunks)
- ✅ Parquet-based caching for fast reloads
- ✅ Preprocessing pipeline (cleaning, resampling, indicators)
- ✅ Data quality validation and reporting

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Strategy   │  │  Backtester  │  │   Optimizer  │    │
│  │  (ORB, etc)  │  │   Engine     │  │   (Optuna)   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│         ┌──────────────────┴──────────────────┐            │
│         │                                      │            │
│  ┌──────▼───────┐                    ┌────────▼────────┐  │
│  │ Data Pipeline│                    │  Paper Trading  │  │
│  │ (Loader,     │                    │     Engine      │  │
│  │  Preprocessor│                    │  (Redis + FIX)  │  │
│  │  Indicators) │                    └─────────────────┘  │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Data Layer** (`src/data/`)
   - `DataLoader`: PostgreSQL tick data retrieval with caching
   - `Preprocessor`: Data cleaning, resampling, feature engineering
   - `Indicators`: Technical indicators (ATR, ADX, Volume MA, etc.)

2. **Strategy Layer** (`src/strategy/`)
   - `BaseStrategy`: Abstract strategy interface
   - `IntradayBase`: Intraday strategy base with session management
   - `ORBStrategy`: Opening Range Breakout implementation
   - Plugin system for custom strategies

3. **Engine Layer** (`src/engine/`)
   - `Backtester`: Event-driven backtesting engine
   - `Account`: Portfolio and position management
   - `SimBroker`: Order execution simulation
   - `EquityTracker`: Equity curve tracking

4. **Paper Trading Layer** (`src/paper/`)
   - `PaperEngine`: Live/dry-run/sim paper trading orchestrator
   - `RedisFeed`: Real-time market data from Redis
   - `SimFeed`: Historical data replay
   - `OrderManager`: Order submission and tracking
   - `Tracker`: Account and position tracking
   - `Reconciler`: Position reconciliation with broker

5. **Optimization Layer** (`src/optimization/`)
   - Optuna integration with custom objectives
   - Walk-forward analysis framework
   - Parameter space management

6. **Metrics Layer** (`src/metrics/`)
   - Performance metrics calculation
   - Trade analysis
   - Visualization and reporting

---

## Installation

### Prerequisites

- Python 3.13 or higher
- PostgreSQL (for historical data storage)
- Redis (for live paper trading market data)
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/BuyHigh-SellLow.git
cd BuyHigh-SellLow
```

2. **Create virtual environment**

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
# Core dependencies only
pip install -e .

# With optimization support
pip install -e ".[optimization]"

# With paper trading support
pip install -e ".[paper]"

# With visualization
pip install -e ".[viz]"

# Development environment (includes testing tools)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

4. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: PostgreSQL connection
- `MARKET_REDIS_HOST`, `MARKET_REDIS_PORT`: Redis for live market data (paper trading only)
- `PAPER_USERNAME`, `PAPER_PASSWORD`, `PAPER_REST_BASE_URL`: Broker credentials (paper trading only)
- `SOCKET_CONNECT_HOST`, `SOCKET_CONNECT_PORT`: FIX connection (live paper trading only)

5. **Verify installation**

```bash
# Run tests
pytest tests/

# Check data loader
python -m src.run_data_loader --help
```

---

## Quick Start

### 1. Load Data

```bash
# Fetch tick data from database for VN30F1M
python -m src.run_data_loader fetch --symbol VN30F1M --start 2024-01-01 --end 2024-12-31

# View data statistics
python -m src.run_data_loader stats --symbol VN30F1M --start 2024-01-01 --end 2024-12-31 --freq 5min
```

### 2. Run Backtest

```bash
# Run backtest with default ORB parameters
python -m src.run_backtest --strategy orb --symbol VN30F1M --start 2024-01-01 --end 2024-06-30

# Run with custom config
python -m src.run_backtest --strategy orb --config config/strategy_params/orb_custom.json
```

### 3. Optimize Strategy

```bash
# Run optimization with 100 trials
python -m src.run_optimize --strategy orb --symbol VN30F1M --trials 100 --start 2024-01-01 --end 2024-06-30

# Results saved to: results/optimization/orb_optimized_params.json
```

### 4. Validate with Walk-Forward

```bash
# Run walk-forward analysis
python -m src.run_walk_forward --strategy orb --symbol VN30F1M --window-months 3 --step-months 1
```

### 5. Paper Trade

```bash
# SIM mode (historical replay, no external connections)
python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim --sample 100

# DRY-RUN mode (live data, orders logged only)
python -m src.run_paper_trade --strategy orb --symbol VN30F1M --dry-run

# LIVE mode (live data + order execution)
python -m src.run_paper_trade --strategy orb --symbol VN30F1M
```

---

## Usage

### Data Management

The data pipeline handles tick-to-bar conversion with caching for performance.

#### Fetch Historical Data

```bash
# Fetch data for specific period
python -m src.run_data_loader fetch --symbol VN30F1M --start 2024-01-01 --end 2024-12-31

# Force refresh (bypass cache)
python -m src.run_data_loader fetch --symbol VN30F1M --start 2024-01-01 --end 2024-12-31 --force-refresh
```

#### View Data Statistics

```bash
# Show data quality and statistics
python -m src.run_data_loader stats --symbol VN30F1M --start 2024-01-01 --end 2024-12-31 --freq 5min
```

**Data Flow:**
1. Tick data fetched from PostgreSQL in 30-day chunks
2. Cached as Parquet files in `data/cache/`
3. Preprocessed: cleaning, resampling, indicator calculation
4. Ready for backtesting or paper trading

### Backtesting

Run historical simulations with realistic order execution.

#### Basic Backtest

```bash
python -m src.run_backtest --strategy orb --symbol VN30F1M --start 2024-01-01 --end 2024-06-30
```

#### With Custom Parameters

```bash
python -m src.run_backtest --strategy orb --config config/strategy_params/orb_custom.json
```

#### Output

Results saved to `results/backtest_{timestamp}/`:
- `trades.parquet`: Trade-by-trade details
- `equity_curve.parquet`: Equity snapshots
- `metrics.json`: Performance metrics
- `*.png`: Visualization charts

**Key Metrics:**
- Total Return, Annualized Return, CAGR
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Longest Drawdown
- Win Rate, Profit Factor
- Average Win/Loss

### Optimization

Find optimal strategy parameters using Optuna.

#### Run Optimization

```bash
# Basic optimization
python -m src.run_optimize --strategy orb --symbol VN30F1M --trials 100

# With specific date range
python -m src.run_optimize --strategy orb --symbol VN30F1M --trials 100 --start 2024-01-01 --end 2024-06-30

# Parallel execution (4 workers)
python -m src.run_optimize --strategy orb --trials 100 --n-jobs 4
```

#### Optimization Objective

The optimizer uses a composite scoring function:

```
Score = Sharpe - |0.1 × Max Drawdown| - |0.1 × Trades/1000|
```

- Maximizes risk-adjusted returns (Sharpe)
- Penalizes large drawdowns
- Penalizes excessive trading (overfitting)

**Safeguards:**
- Minimum 100 trades required (prevents low-frequency noise)
- Negative Sharpe fallback to Total Return / 100

#### Output

Optimized parameters saved to `results/optimization/orb_optimized_params.json`

### Walk-Forward Analysis

Validate strategy robustness across different market regimes.

#### Run Walk-Forward

```bash
# 3-month training window, 1-month test window
python -m src.run_walk_forward --strategy orb --symbol VN30F1M --window-months 3 --step-months 1

# Anchored walk-forward (expanding window)
python -m src.run_walk_forward --strategy orb --anchored
```

#### Process

1. Split data into overlapping train/test windows
2. Optimize on training window
3. Test on out-of-sample window
4. Aggregate results across all windows

#### Output

Results saved to `results/walk_forward_{timestamp}/`:
- Per-window performance metrics
- Aggregated statistics
- Stability analysis

### Paper Trading

Run strategies in real-time or simulation mode.

#### Operating Modes

**SIM Mode** (Historical Replay)
```bash
# Replay last 100 bars
python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim --sample 100

# Replay specific date range
python -m src.run_paper_trade --strategy orb --sim --sim-start 2024-01-01 --sim-end 2024-01-31
```

**DRY-RUN Mode** (Live Data, No Execution)
```bash
python -m src.run_paper_trade --strategy orb --symbol VN30F1M --dry-run
```

**LIVE Mode** (Live Data + Execution)
```bash
python -m src.run_paper_trade --strategy orb --symbol VN30F1M
```

#### Features

- Real-time bar aggregation from Redis tick stream
- Data quality validation (staleness, gaps, update frequency)
- Position reconciliation with broker
- Session-aware trading (morning/afternoon sessions)
- Automatic position flattening at session end
- Comprehensive logging and statistics

#### Output

Session results saved to `results/paper_{timestamp}/`:
- `trades.parquet`: Executed trades
- `equity_curve.parquet`: Equity snapshots
- `session_metrics.json`: Performance metrics

---

## Strategy Development

### Creating a Custom Strategy

1. **Create strategy file**: `src/strategy/my_strategy.py`

```python
from src.strategy.intraday_base import IntradayBase
from src.strategy.signal import Signal, TradeSignal

class MyStrategy(IntradayBase):
    """My custom strategy."""

    def __init__(self, **params):
        super().__init__(**params)
        # Initialize strategy-specific parameters
        self.my_param = params.get("my_param", 1.0)

    def generate_signal(self, bar: dict, timestamp) -> TradeSignal:
        """Generate trading signal for current bar."""
        # Your strategy logic here
        if self._should_buy(bar):
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=bar["close"],
                stop_loss=bar["close"] - self.atr * 2,
                take_profit=bar["close"] + self.atr * 3,
                ord_type="MARKET",
                reason="My buy condition"
            )

        return TradeSignal(signal=Signal.HOLD)

    def _should_buy(self, bar: dict) -> bool:
        # Your entry logic
        return False
```

2. **Create plugin file**: `src/strategy/my_strategy_plugin.py`

```python
from src.strategy.my_strategy import MyStrategy
from src.strategy.strategy_registry import register_strategy_plugin

register_strategy_plugin(
    name="mystrat",
    strategy_class=MyStrategy,
    default_params={
        "my_param": 1.0,
        "atr_period": 14,
        # ... other parameters
    }
)
```

3. **Use your strategy**

```bash
python -m src.run_backtest --strategy mystrat --symbol VN30F1M
```

### Strategy Interface

All strategies must implement:

- `generate_signal(bar, timestamp) -> TradeSignal`: Generate trading signal
- `on_fill(fill_price, qty, side, timestamp)`: Handle order fills
- `on_bar(bar)`: Process new bar (optional)

### Built-in Strategies

#### Opening Range Breakout (ORB)

Trades breakouts of the opening range established in the first N minutes of each session.

**Key Parameters:**
- `orb_minutes`: Opening range duration (default: 15)
- `atr_period`: ATR calculation period (default: 14)
- `atr_tp_multiplier`: Take profit distance (default: 2.0)
- `atr_sl_multiplier`: Stop loss distance (default: 1.5)
- `breakout_buffer`: Breakout confirmation buffer (default: 0.1)
- `min_range_atr`: Minimum range size filter (default: 0.5)
- `max_range_atr`: Maximum range size filter (default: 3.0)
- `long_only`: Only take long positions (default: false)
- `use_volume_filter`: Require above-average volume (default: false)
- `use_adx_filter`: Require trending market (default: false)

---

## Configuration

### Strategy Parameters

Strategy parameters are defined in JSON files in `config/strategy_params/`.

**Example**: `config/strategy_params/orb_default.json`

```json
{
  "strategy": {
    "resample_freq": "5min",
    "orb_minutes": 15,
    "atr_period": 14,
    "atr_tp_multiplier": 2.0,
    "atr_sl_multiplier": 1.5,
    "breakout_buffer": 0.1,
    "use_range_sl": true,
    "min_range_atr": 0.5,
    "max_range_atr": 3.0,
    "long_only": false,
    "use_volume_filter": false,
    "use_adx_filter": false,
    "adx_min": 20
  },
  "risk": {
    "min_position_size": 1,
    "max_position_size": 10,
    "risk_per_trade_pct": 1.0,
    "max_daily_loss": 0.02,
    "use_trailing_stop": true,
    "trailing_atr_multiplier": 2.0
  }
}
```

### Environment Configuration

See `.env.example` for all available environment variables.

**Key Settings:**

**Database:**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

**Paper Trading:**
- `MARKET_REDIS_HOST`, `MARKET_REDIS_PORT`: Market data source
- `PAPER_USERNAME`, `PAPER_PASSWORD`: Broker credentials
- `SOCKET_CONNECT_HOST`, `SOCKET_CONNECT_PORT`: FIX connection

**Runtime Toggles:**
- `PAPER_ENABLE_DB_BAR_FALLBACK`: Fetch closed bars from DB when live data missing
- `PAPER_CLOSE_ON_SHUTDOWN`: Close positions on shutdown
- `PAPER_ENTRY_CUTOFF_SECONDS`: Block entries before session end
- `PAPER_FORCE_FLAT_ON_SESSION_CLOSE`: Force flat at session boundary

**Logging:**
- `LOG_FORMAT`: "text" or "json"
- `LOG_CAPTURE_ALL`: Capture all loggers (0/1)

---

## Testing

The project has comprehensive test coverage with 396 tests (100% pass rate).

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/paper/test_redis_feed.py

# Run with coverage report
pytest --cov=src --cov-report=html

# Run paper trading tests only
pytest tests/paper/
```

### Test Categories

**Core Tests** (215 tests)
- Account tracking and reconciliation
- Order management
- Risk management
- Bar handling
- Signal handling
- Engine orchestration

**Integration Tests** (181 tests)
- Redis market data integration
- Cash accounting
- SessionStats and reporting
- Database persistence
- LIVE/DRY-RUN/SIM modes
- Entry point validation

### Code Quality

```bash
# Lint and format
ruff check .
ruff format .

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

## Project Structure

```
BuyHigh-SellLow/
├── config/                      # Configuration files
│   ├── constants.py            # System constants
│   ├── secrets.py              # Credential management
│   ├── schemas/                # Pydantic schemas
│   └── strategy_params/        # Strategy parameter files
│
├── src/                        # Source code
│   ├── data/                   # Data pipeline
│   │   ├── loader.py          # Data loading from DB
│   │   ├── preprocessor.py    # Data cleaning and resampling
│   │   ├── pipeline.py        # Full data pipeline
│   │   ├── validators.py      # Data quality validation
│   │   └── indicators/        # Technical indicators
│   │
│   ├── engine/                 # Backtesting engine
│   │   ├── backtester.py      # Main backtester
│   │   ├── account/           # Portfolio management
│   │   ├── execution/         # Order execution simulation
│   │   └── session/           # Session management
│   │
│   ├── strategy/               # Strategy framework
│   │   ├── base.py            # Base strategy interface
│   │   ├── intraday_base.py   # Intraday strategy base
│   │   ├── orb.py             # ORB strategy implementation
│   │   ├── signal.py          # Signal definitions
│   │   └── strategy_registry.py # Strategy plugin system
│   │
│   ├── paper/                  # Paper trading system
│   │   ├── engine.py          # Paper trading engine
│   │   ├── account/           # Account tracking
│   │   ├── execution/         # Order management
│   │   ├── feeds/             # Market data feeds
│   │   ├── handlers/          # Event handlers
│   │   ├── bar_aggregator.py  # Tick-to-bar aggregation
│   │   ├── data_quality.py    # Data quality validation
│   │   └── stats.py           # Session statistics
│   │
│   ├── optimization/           # Optimization framework
│   │   ├── optuna_search.py   # Optuna integration
│   │   ├── walk_forward.py    # Walk-forward analysis
│   │   └── scoring.py         # Objective functions
│   │
│   ├── metrics/                # Performance metrics
│   │   ├── metrics.py         # Metrics calculator
│   │   ├── trade_metrics.py   # Trade analysis
│   │   └── plotter.py         # Visualization
│   │
│   ├── database/               # Database layer
│   │   ├── connection.py      # Connection management
│   │   ├── query.py           # Query builders
│   │   └── data_service.py    # Data service
│   │
│   ├── utils/                  # Utilities
│   │   └── logger.py          # Logging setup
│   │
│   ├── run_data_loader.py     # Data loading CLI
│   ├── run_backtest.py        # Backtesting CLI
│   ├── run_optimize.py        # Optimization CLI
│   ├── run_walk_forward.py    # Walk-forward CLI
│   ├── run_paper_trade.py     # Paper trading CLI
│   └── run_account_check.py   # Account inspector CLI
│
├── tests/                      # Test suite
│   ├── paper/                 # Paper trading tests
│   ├── engine/                # Backtesting tests
│   └── data/                  # Data pipeline tests
│
├── data/                       # Data storage
│   └── cache/                 # Parquet cache files
│
├── results/                    # Output directory
│   ├── backtest_*/            # Backtest results
│   ├── optimization_*/        # Optimization results
│   ├── walk_forward_*/        # Walk-forward results
│   └── paper_*/               # Paper trading results
│
├── logs/                       # Log files
├── reports/                    # Generated reports
├── docs/                       # Documentation
├── .env.example               # Environment template
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

---

## Performance

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 10 GB (for data cache)

**Recommended:**
- CPU: 4+ cores (for parallel optimization)
- RAM: 8+ GB
- Storage: 50+ GB
- SSD for data cache

### Benchmarks

**Data Loading:**
- Tick data fetch: ~30 days in 2-5 seconds (cached)
- Preprocessing: ~1 year in 10-20 seconds

**Backtesting:**
- 1 year, 5-min bars: ~2-5 seconds
- 1 year, 1-min bars: ~5-10 seconds

**Optimization:**
- 100 trials: ~10-30 minutes (depends on parameter space)
- Parallel execution: ~4x speedup with 4 cores

**Paper Trading:**
- Bar aggregation latency: <100ms
- Order submission latency: <50ms (FIX)

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Run tests**: `pytest`
5. **Run linters**: `ruff check . && ruff format .`
6. **Commit**: `git commit -m "Add my feature"`
7. **Push**: `git push origin feature/my-feature`
8. **Create Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests with coverage
pytest --cov=src --cov-report=html

# Check code quality
ruff check .
mypy src/
```

### Code Style

- Follow PEP 8 (enforced by Ruff)
- Use type hints (checked by mypy)
- Write docstrings for public APIs
- Add tests for new features
- Keep functions focused and small

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **ALGOTRADE** - Algorithmic Trading Theory and Practice book for ORB strategy reference
- **Optuna** - Hyperparameter optimization framework
- **pandas** - Data manipulation and analysis
- **pytest** - Testing framework

---

## Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [https://github.com/yourusername/BuyHigh-SellLow/issues](https://github.com/yourusername/BuyHigh-SellLow/issues)
- **Email**: your.email@example.com

---

## Disclaimer

This software is for educational and research purposes only. Trading futures involves substantial risk of loss. Past performance is not indicative of future results. Always conduct your own research and consult with a qualified financial advisor before trading.

---

**Built with ❤️ for algorithmic traders**
