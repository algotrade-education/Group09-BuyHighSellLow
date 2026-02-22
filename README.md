# Algorithm Name

## Abstract

## Introduction

## Trading Hypothesis

### Target Market

### Entry Conditions

#### Buy Signal (Bullish Hypothesis)

#### Sell Signal (Bearish Hyopothesis)

### Indicators Used

### Order Execution

### Exit Conditions

---

## Installation

### Prerequisites

- Python 3.12+
- pip or conda

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/rlukas2/BuyHigh-SellLow.git
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
pip install -r requirements.txt
```

4. **Set up environment variables** (for database access)

```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. **Load data** (requires database connection)

```bash
python -m src.run_data_loader
```

### Usage

#### Fetch/Load Data

```bash
python -m src.run_data_loader --mode [fetch|load] --sample [is|os] --contract [contract_name]

# Example: Fetch data for contract "VN30F1M"
python -m src.run_data_loader --mode fetch --contract VN30F1M
```

#### Run Backtest

```bash
python -m src.run_backtest --sample [is|os] --contract [contract_name] --config [config_file]
# Example: Run in-sample backtest for "VN30F1M"
python -m src.run_backtest --sample is --contract VN30F1M
# Example: Run out-of-sample backtest with optimized parameters
python -m src.run_backtest --sample os --contract VN30F1M --config config/strategy_params/optimized_params.json
```

#### Run Optimization

```bash
python -m src.run_optimization --contract [contract_name]
```

#### Run Walk-Forward Analysis

```bash
python -m src.run_walk_forward --contract [contract_name]
```

---

## Data

The strategy relies on high-quality tick-by-tick market data, specifically for the **VN30F1M** futures contract. Data management is handled by the `DataLoader` class, ensuring efficient retrieval and processing.

### Data Collection

By default, the system fetches **VN30F1M** tick-by-tick data from a PostgreSQL database. The data pipeline is designed for reliability and performance:

1.  **Database Querying**:
    -   **Matched Data**: The core query fetches trade execution data (`price`, `datetime`) from the `quote.matched` table. It joins with `quote.futurecontractcode` to filter for the specific future contract and `quote.total` to retrieve cumulative volume (`quantity`) at each tick.
    -   **Chunking**: To prevent timeouts and memory issues with large datasets, data is fetched in **30-day chunks**. These chunks are then concatenated into a single DataFrame.
    -   **Close Data**: Daily close prices are fetched separately from `quote.close` to provide reference points if needed.

2.  **Caching Mechanism**:
    -   The system implements a robust caching layer using **Parquet** files (`.parquet`) to minimize database load.
    -   **Load**: Before querying the database, the `DataLoader` checks the `data/cache/` directory for an existing file matching the request (contract name, start date, end date). If found, data is loaded instantly from the cache.
    -   **Save**: If data is fetched from the database, it is automatically saved to the cache for future runs.
    -   **Format**: Cache filenames follow a consistent pattern: `{contract}_{start}_{end}.parquet`.

### Data Processing

Raw tick data undergoes a rigorous preprocessing pipeline before being used in backtests. This is handled by the `Preprocessor` class:

1.  **Cleaning**:
    -   **Deduplication**: Duplicate entries are removed.
    -   **Missing Values**: Price columns are forward-filled (`ffill`) to handle gaps. Rows with missing critical timestamps are dropped.

2.  **Feature Engineering**:
    -   **Volume Derivation**: The raw `quantity` column represents cumulative volume. The preprocessor calculates per-tick `volume` by taking the difference of `quantity` between consecutive ticks, handling daily resets automatically.
    -   **Resampling**: Convert tick-by-tick data into OHLC (Open-High-Low-Close) bars at the specified frequency (e.g., `5min`, `15min`, `1h`) to standardize the timeframe for strategy execution.

3.  **Filtering**:
    -   **Trading Hours**: Data is filtered to strictly adhere to active trading hours (e.g., 09:00 - 14:30), removing pre-market or post-market noise unless ATC (At-The-Close) sessions are explicitly requested.

4.  **Indicator Calculation**:
    -   Computed on the resampled data:
        -   **SMA & Slope**: Simple Moving Average (20-period) and its 1-period slope.
        -   **Bollinger Bands**: 20-period, 2.0 standard deviation.
        -   **Volume MA**: 20-period moving average of volume.
        -   **RSI**: 14-period Relative Strength Index.
        -   **ADX**: 14-period Average Directional Index (trend strength).

5.  **Validation**:
    -   The pipeline concludes with a validation step to ensure datetime continuity and data integrity before passing the dataset to the backtester.

## In-Sample Backtesting

### Parameters

### Results

## Optimization

### Parameters

### Results

## Out-of-sample Backtesting

### Parameters

### Results
