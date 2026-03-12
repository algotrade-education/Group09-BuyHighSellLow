from datetime import datetime, timedelta

from src.data.preprocessor import Preprocessor
from src.database.data_service import fetch_and_merge_data
from src.utils.cli_helpers import load_orb_config_context

config, strategy_params, resample_freq = load_orb_config_context(
    "config/strategy_params/orb_optuna_20260306.json"
)

end_date = datetime.now()
start_date = end_date - timedelta(days=5)
print(
    f"Fetching from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
)

raw = fetch_and_merge_data(
    "VN30F1M", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
)
print(f"fetch_and_merge_data returned {len(raw)} rows")
print(raw.head(3))
print(raw.tail(3))
print("Columns:", raw.columns.tolist())

preprocessor = Preprocessor(
    sma_period=20,
    bb_std=2.0,
    atr_period=14,
    volume_ma_period=20,
)

df = preprocessor.clean_data(raw)
print(f"After clean_data: {len(df)} rows")

df = preprocessor._derive_volume(df, copy=False)
print(f"After _derive_volume: {len(df)} rows. Columns: {df.columns.tolist()}")

df = preprocessor.resample_to_ohlc(df, freq=resample_freq)
print(f"After resample_to_ohlc: {len(df)} rows. Columns: {df.columns.tolist()}")

df = preprocessor.filter_trading_hours(df, include_atc=True)
print(f"After filter_trading_hours: {len(df)} rows")

bb_period = int(strategy_params.get("bb_period", 20))
bb_std = float(strategy_params.get("bb_std", 2.0))
kc_period = int(strategy_params.get("kc_period", 20))
kc_mult = float(strategy_params.get("kc_mult", 1.5))
atr_period = int(strategy_params.get("atr_period", 14))
mom_period = int(strategy_params.get("mom_period", 12))
vol_ma_period = int(strategy_params.get("vol_ma_period", 20))

df = preprocessor.add_atr(df, period=atr_period, copy=True)
print(f"After add_atr: {len(df.dropna())} rows valid out of {len(df)}")
df = preprocessor.add_bollinger_bands(df, period=bb_period, std_dev=bb_std, copy=False)
print(f"After add_bollinger_bands: {len(df.dropna())} rows valid out of {len(df)}")
df = preprocessor.add_volume_ma(df, period=vol_ma_period, copy=False)
print(f"After add_volume_ma: {len(df.dropna())} rows valid out of {len(df)}")
df = preprocessor.add_keltner_channels(
    df, ema_period=kc_period, atr_period=atr_period, multiplier=kc_mult, copy=False
)
print(f"After add_keltner_channels: {len(df.dropna())} rows valid out of {len(df)}")
df = preprocessor.add_momentum(df, period=mom_period, copy=False)
print(f"After add_momentum: {len(df.dropna())} rows valid out of {len(df)}")
df = preprocessor.add_session_vwap(df, copy=False)
print(f"After add_session_vwap: {len(df.dropna())} rows valid out of {len(df)}")

df.dropna(inplace=True)
print(f"Final valid rows: {len(df)}")
