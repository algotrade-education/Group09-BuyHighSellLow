"""
Technical indicators for trading strategies.

This module provides stateful indicator implementations that can be updated
bar-by-bar for real-time or backtesting scenarios.

Available indicators:
- WilderATR: Average True Range using Wilder's smoothing
- WilderADX: Average Directional Index with DI+/DI-
- VolumeMA: Simple moving average of volume

Usage:
    from data.indicators import WilderATR, WilderADX, VolumeMA

    atr = WilderATR(period=14)
    adx = WilderADX(period=14)
    volume_ma = VolumeMA(period=20)

    # Update with bar data
    atr_value = atr.update(high=100, low=95, close=98)
    adx_value = adx.update(high=100, low=95, close=98)
    vol_ma = volume_ma.update(volume=1000000)
"""

from src.data.indicators.adx import WilderADX
from src.data.indicators.atr import WilderATR
from src.data.indicators.base import IndicatorBase
from src.data.indicators.registry import (
    IndicatorRegistry,
    IndicatorSpec,
)
from src.data.indicators.volume_ma import VolumeMA

__all__ = [
    "IndicatorBase",
    "WilderATR",
    "WilderADX",
    "VolumeMA",
    "IndicatorRegistry",
    "IndicatorSpec",
]
