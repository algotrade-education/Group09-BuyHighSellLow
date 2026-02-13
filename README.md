# VN30F Trading Strategy

A systematic trading strategy for VN30 futures (VN30F1M) implementing the **Bollinger Mean Reversion** hypothesis with comprehensive backtesting, optimization, and walk-forward analysis.

## 📋 Table of Contents

- [Overview](#-overview)
- [Strategy Hypothesis](#-strategy-hypothesis)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Testing](#-testing)
- [Results](#-results)
- [Contributing](#-contributing)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🎯 Overview

This project implements a **Bollinger Mean Reversion Strategy** for VN30 futures based on the SMA(20) slope hypothesis.

---

## 💡 Strategy Hypothesis

> In a structured trending market, prices oscillate around a fair value mean (SMA 20). When price retraces to touch the Middle Band during a trend, it represents a high-probability "value entry" point.

### Entry Rules

| Direction | Condition 1 | Condition 2 | Condition 3 |
|-----------|-------------|-------------|-------------|
| **LONG** | SMA(20) slope > 0 | Price Low ≤ SMA(20) | Price Close > SMA(20) |
| **SHORT** | SMA(20) slope < 0 | Price High ≥ SMA(20) | Price Close < SMA(20) |

### Exit Rules

| Exit Type | LONG Position | SHORT Position |
|-----------|---------------|----------------|
| **Take Profit** | Close ≥ Upper BB | Close ≤ Lower BB |
| **Stop Loss** | Close < SMA(20) | Close > SMA(20) |
| **EOD Close** | At ATC (14:30-14:45) | At ATC (14:30-14:45) |

### Execution

- **Order Type**: Limit order at SMA(20) price level
- **Position Size**: 1 contract (fixed)
- **Commission**: 0.015% per side
- **Slippage**: 0.5 index points

### Limitations

The strategy is expected to underperform in:
- Range-bound markets
- Highly volatile markets where SMA does not act as reliable equilibrium

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/rlukas2/vn30_strategy.git
cd vn30_strategy
```

2. **Create virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (for database access)

```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. **Load data** (requires database connection)

```bash
python -m src.data.loader
```

---

## ⚡ Quick Start

### 1. List available parameter configurations

```bash
python main.py --mode list
```

### 2. Run in-sample backtest

```bash
python main.py --mode backtest --params default --sample is
```

### 3. Run out-of-sample validation

```bash
python main.py --mode backtest --params default --sample os
```

### 4. Optimize parameters

```bash
python main.py --mode optimize --sample is
```

### 5. Walk-forward analysis

```bash
python main.py --mode walk_forward --sample is
```

---

## 📖 Usage

### Command-Line Interface

```bash
python main.py --mode <MODE> [OPTIONS]
```

**Modes:**

| Mode | Description |
|------|-------------|
| `list` | List available parameter configs |
| `fetch` | Fetch data from database to CSV |
| `backtest` | Run backtest with specified params |
| `optimize` | Grid search optimization |
| `walk_forward` | Walk-forward analysis |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--params` | Parameter config name | `default` |
| `--sample` | Data sample (`is` or `os`) | `is` |
| `--capital` | Initial capital | `100000` |
| `--contract` | Contract type | `VN30F1M` |

**Examples:**

```bash
# Backtest with default params on in-sample data
python main.py --mode backtest --params default --sample is

# Backtest with optimized params on out-of-sample
python main.py --mode backtest --params optimized_20260207 --sample os

# Optimize with 50k capital
python main.py --mode optimize --capital 50000
```

---

## ⚙️ Configuration

### JSON Parameter Files

Parameters are stored in `config/strategy_params/*.json`:

**Example: `config/strategy_params/default.json`**

```json
{
    "name": "Default Parameters",
    "description": "Bollinger Mean Reversion strategy with SMA(20) slope",
    "strategy": {
        "sma_period": 20,
        "bb_std": 2.0,
        "slope_lookback": 1,
        "position_size": 1
    },
    "risk": {
        "commission_rate": 0.00015,
        "slippage_points": 0.5,
        "max_positions": 1,
        "close_at_eod": true
    },
    "trading_hours": {
        "start": "09:00:00",
        "end": "14:30:00",
        "atc_start": "14:30:00",
        "atc_end": "14:45:00"
    }
}
```

### Creating Custom Configs

1. Copy default config:
```bash
cp config/strategy_params/default.json config/strategy_params/aggressive.json
```

2. Edit parameters in the new file

3. Test your config:
```bash
python main.py --mode backtest --params aggressive
```

---

## 🧪 Testing

Run the full test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_indicators.py -v
pytest tests/test_strategy.py -v
pytest tests/test_backtester.py -v
```

### Test Coverage

| Module | Description |
|--------|-------------|
| `test_indicators.py` | SMA, Bollinger Bands calculations |
| `test_strategy.py` | Signal generation logic |
| `test_position.py` | Order and position management |
| `test_metrics.py` | Performance metric calculations |
| `test_backtester.py` | Integration tests |

---

## 📊 Results

Results are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `equity_curve_*.csv` | Equity over time |
| `trades_*.csv` | Trade-by-trade log |
| `optimization_grid_*.csv` | Grid search results |
| `walk_forward_*.csv` | Walk-forward results |

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

---

## 📝 Requirements

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pytest>=7.0.0
pytest-cov>=4.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
```

---

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

---

## 📞 Contact

**Author:** Tuan Ngo-Hoang - 22125115  
**Email:** nhtuan22@apcs.fitus.edu.vn | rickielukas@gmail.com  
**GitHub:** [@rlukas2](https://github.com/rlukas2)

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading futures involves substantial risk of loss
- This is NOT financial advice
- Always test strategies thoroughly before live trading
- Use proper risk management
- Consult a financial advisor

**USE AT YOUR OWN RISK.**
