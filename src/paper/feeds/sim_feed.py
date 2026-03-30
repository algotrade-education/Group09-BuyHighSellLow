"""Simulation feed for historical bar-by-bar replay.

Replays a historical OHLCV DataFrame through the indicator pipeline, emitting
bars one at a time to simulate live trading conditions.

Key features:
- Pre-calculated indicator support: uses existing indicator columns if present
- Rolling history buffer: maintains warmup period for indicator calculation
- Chronological order guarantee: emits bars in input DataFrame order
- Configurable replay speed: sleep between bars for realistic timing
- Warmup silence: skips emitting until history buffer reaches atr_period + 1
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.paper.feeds.base import FeedBase

logger = logging.getLogger(__name__)


class SimFeed(FeedBase):
    """Historical bar replay feed for simulation mode.

    Replays a historical DataFrame bar-by-bar, either using pre-calculated
    indicators or computing them on-the-fly via a rolling history buffer.

    Supports two modes:
    1. Pre-calculated indicators: DataFrame has indicator columns → emit directly
    2. On-the-fly calculation: Maintain rolling buffer, compute via pipeline
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline: Any | None = None,
        atr_period: int = 14,
        speed: float = 0.0,
    ) -> None:
        """Initialize SimFeed.

        Args:
            df: Historical OHLCV DataFrame with datetime, open, high, low, close, volume.
                May optionally include pre-calculated indicator columns.
            pipeline: Optional DataPipeline instance for indicator computation.
                      If None and df lacks indicators, bars will be emitted without indicators.
            atr_period: ATR period for warmup calculation (default 14).
            speed: Seconds to sleep between bars (0 = max throughput).
        """
        self._df = df
        self._pipeline = pipeline
        self._atr_period = atr_period
        self._speed = speed

        self._callback: Callable[[dict], None] | None = None
        self._running = False
        self._replay_task: asyncio.Task | None = None

        # Check if DataFrame has pre-calculated indicators
        self._has_indicators = self._detect_indicators(df)

        # Rolling history buffer for on-the-fly indicator calculation
        self._history: list[dict] = []
        self._warmup_threshold = atr_period + 1

    def _detect_indicators(self, df: pd.DataFrame) -> bool:
        """Detect if DataFrame has pre-calculated indicator columns.

        Looks for common indicator column patterns (e.g. atr_14, adx_14, ema_20).

        Args:
            df: DataFrame to check.

        Returns:
            True if indicator columns detected, False otherwise.
        """
        if df.empty:
            return False

        # Common indicator column patterns
        indicator_patterns = ["atr_", "adx_", "ema_", "sma_", "rsi_", "macd_", "bb_"]

        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in indicator_patterns):
                logger.info("Detected pre-calculated indicators in DataFrame")
                return True

        return False

    async def subscribe(self, symbol: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to simulated market data.

        Starts the bar-by-bar replay task.

        Args:
            symbol: Symbol (ignored in sim mode, for interface compatibility).
            callback: Callback to invoke when a new bar is ready.
        """
        if self._running:
            logger.warning("SimFeed already running, ignoring duplicate subscribe")
            return

        self._callback = callback
        self._running = True

        # Start replay task
        self._replay_task = asyncio.create_task(self._replay())

        logger.info("SimFeed started replay for %d bars", len(self._df))

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from simulated market data.

        Args:
            symbol: Symbol (ignored in sim mode).
        """
        if not self._running:
            return

        self._running = False

        # Cancel replay task
        if self._replay_task is not None:
            self._replay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._replay_task
            self._replay_task = None

        logger.info("SimFeed replay stopped")

    async def close(self) -> None:
        """Close the feed and clean up resources."""
        if self._running:
            await self.unsubscribe("")

        logger.info("SimFeed closed")

    async def wait_for_completion(self) -> None:
        """Wait until the replay task finishes.

        Returns immediately if no replay is running.
        """
        if self._replay_task is not None:
            try:
                await self._replay_task
            except asyncio.CancelledError:
                logger.info("[SimFeed] Replay cancelled")
            except Exception:
                logger.exception("[SimFeed] Replay failed")

    async def _replay(self) -> None:
        """Replay historical bars one at a time.

        Workflow:
        1. Check if DataFrame is empty → log warning and return
        2. If has_indicators → emit bars directly (skip re-calculation)
        3. Otherwise → maintain rolling history buffer, compute via pipeline
        4. Warmup: skip emitting until len(history) >= atr_period + 1
        5. Between bars: await asyncio.sleep(speed)
        """
        try:
            # Validate DataFrame
            if self._df.empty:
                logger.warning("SimFeed: empty DataFrame, no bars to replay")
                return

            required = {"datetime", "open", "high", "low", "close"}
            missing = required - set(self._df.columns)
            if missing:
                logger.error("SimFeed: DataFrame missing required columns: %s", missing)
                return

            bars_emitted = 0

            # Mode 1: Pre-calculated indicators → emit directly
            if self._has_indicators:
                logger.info("SimFeed: using pre-calculated indicators")

                for _, row in self._df.iterrows():
                    if not self._running:
                        break

                    bar = row.to_dict()

                    # Ensure datetime is datetime object
                    if isinstance(bar["datetime"], pd.Timestamp):
                        bar["datetime"] = bar["datetime"].to_pydatetime()

                    # Emit bar directly
                    self._emit_bar(bar)
                    bars_emitted += 1

                    # Sleep between bars
                    if self._speed > 0:
                        await asyncio.sleep(self._speed)

            # Mode 2: On-the-fly indicator calculation
            else:
                logger.info("SimFeed: computing indicators on-the-fly")

                for _, row in self._df.iterrows():
                    if not self._running:
                        break

                    bar = row.to_dict()

                    # Ensure datetime is datetime object
                    if isinstance(bar["datetime"], pd.Timestamp):
                        bar["datetime"] = bar["datetime"].to_pydatetime()

                    # Add to history buffer
                    self._history.append(bar)

                    # Warmup: skip emitting until threshold reached
                    if len(self._history) < self._warmup_threshold:
                        logger.debug(
                            "SimFeed warmup: %d/%d bars",
                            len(self._history),
                            self._warmup_threshold,
                        )

                        # Sleep even during warmup to maintain timing
                        if self._speed > 0:
                            await asyncio.sleep(self._speed)

                        continue

                    # Compute indicators on history buffer
                    enriched_bar = self._enrich_bar(bar)

                    # Emit enriched bar
                    if enriched_bar is not None:
                        self._emit_bar(enriched_bar)
                        bars_emitted += 1

                    # Sleep between bars
                    if self._speed > 0:
                        await asyncio.sleep(self._speed)

            logger.info("SimFeed replay complete: %d bars emitted", bars_emitted)

        except asyncio.CancelledError:
            logger.debug("SimFeed replay cancelled")
            raise

        except Exception:
            logger.exception("SimFeed replay failed")
            raise

    def _enrich_bar(self, bar: dict) -> dict | None:
        """Compute indicators on history buffer and return enriched latest bar.

        Args:
            bar: Raw bar dict without indicators.

        Returns:
            Enriched bar dict with indicators, or None if enrichment fails.
        """
        if self._pipeline is None:
            # No pipeline → return raw bar
            return bar

        try:
            # Convert history to DataFrame
            history_df = pd.DataFrame(self._history)

            # Run pipeline to compute indicators
            enriched_df = self._pipeline.run(history_df)

            # Update history buffer with enriched data
            self._history = enriched_df.to_dict(orient="records")

            # Return enriched latest bar
            return self._history[-1]

        except Exception:
            logger.exception("SimFeed indicator enrichment failed for bar %s", bar.get("datetime"))
            return None

    def _emit_bar(self, bar: dict) -> None:
        """Emit a bar via the registered callback.

        Args:
            bar: Bar dict to emit.
        """
        if self._callback is None:
            return

        try:
            self._callback(bar)
        except Exception:
            logger.exception("SimFeed callback error for bar %s", bar.get("datetime"))
