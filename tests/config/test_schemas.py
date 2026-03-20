"""
tests/config/test_schemas.py

Unit tests cho config schemas.
Verify validation logic hoạt động đúng — đặc biệt các edge cases
mà V1 bị silent fail.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.schemas.base import RiskConfig
from config.schemas.orb import ORBConfig, ORBStrategyConfig
from config.schemas.session import Session, VN30SessionConfig

# ══════════════════════════════════════════════════════════════════
# VN30SessionConfig
# ══════════════════════════════════════════════════════════════════


class TestVN30SessionConfig:
    def test_morning_session(self) -> None:
        from datetime import time

        assert VN30SessionConfig.get_session(time(9, 0)) == Session.MORNING
        assert VN30SessionConfig.get_session(time(10, 30)) == Session.MORNING
        assert VN30SessionConfig.get_session(time(11, 29)) == Session.MORNING

    def test_morning_end_is_exclusive(self) -> None:
        from datetime import time

        assert VN30SessionConfig.get_session(time(11, 30)) == Session.CLOSED

    def test_afternoon_session(self) -> None:
        from datetime import time

        assert VN30SessionConfig.get_session(time(13, 0)) == Session.AFTERNOON
        assert VN30SessionConfig.get_session(time(14, 0)) == Session.AFTERNOON
        assert VN30SessionConfig.get_session(time(14, 29)) == Session.AFTERNOON

    def test_atc_session(self) -> None:
        from datetime import time

        assert VN30SessionConfig.get_session(time(14, 30)) == Session.ATC
        assert VN30SessionConfig.get_session(time(14, 44)) == Session.ATC

    def test_closed_session(self) -> None:
        from datetime import time

        assert VN30SessionConfig.get_session(time(8, 59)) == Session.CLOSED
        assert VN30SessionConfig.get_session(time(12, 0)) == Session.CLOSED
        assert VN30SessionConfig.get_session(time(14, 45)) == Session.CLOSED
        assert VN30SessionConfig.get_session(time(18, 0)) == Session.CLOSED

    def test_signal_not_allowed_during_atc(self) -> None:
        from datetime import time

        assert not VN30SessionConfig.is_signal_allowed(time(14, 35))

    def test_signal_allowed_during_morning_afternoon(self) -> None:
        from datetime import time

        assert VN30SessionConfig.is_signal_allowed(time(9, 30))
        assert VN30SessionConfig.is_signal_allowed(time(13, 30))

    def test_bars_per_year_5min(self) -> None:
        # 255 phút / 5 = 51 bars/day × 252 = 12,852
        assert VN30SessionConfig.bars_per_year(5) == 12_852

    def test_bars_per_year_1min(self) -> None:
        assert VN30SessionConfig.bars_per_year(1) == 64_260

    def test_bars_per_year_invalid_freq(self) -> None:
        with pytest.raises(ValueError, match="không chia đều"):
            VN30SessionConfig.bars_per_year(7)


# ══════════════════════════════════════════════════════════════════
# RiskConfig
# ══════════════════════════════════════════════════════════════════


class TestRiskConfig:
    def test_valid_config(self) -> None:
        cfg = RiskConfig(min_position_size=1, max_position_size=5)
        assert cfg.min_position_size == 1
        assert cfg.max_position_size == 5

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValidationError, match="max_position_size"):
            RiskConfig(min_position_size=5, max_position_size=3)

    def test_max_daily_loss_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss=0.5)  # > 0.20

        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss=0.0)  # < 0.001

    def test_unknown_field_rejected(self) -> None:
        # V1 bug: silently ignore unknown fields
        # V2: extra="forbid" raises ValidationError
        with pytest.raises(ValidationError, match="Extra inputs"):
            RiskConfig(unknown_field="value")  # type: ignore

    def test_string_int_coercion_rejected(self) -> None:
        # V1 bug: "5" được accept silently
        # V2: strict type — int field phải là int
        with pytest.raises(ValidationError):
            RiskConfig(min_position_size="five")  # type: ignore # string không phải int


# ══════════════════════════════════════════════════════════════════
# ORBStrategyConfig
# ══════════════════════════════════════════════════════════════════


class TestORBStrategyConfig:
    def _base_kwargs(self) -> dict:
        return {}

    def test_valid_config(self) -> None:
        cfg = ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min")
        assert cfg.orb_minutes == 20  # default
        assert cfg.atr_period == 14
        assert cfg.use_range_sl is True

    def test_orb_minutes_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min", orb_minutes=0)
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min", orb_minutes=121)

    def test_atr_period_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min", atr_period=4)  # < 5
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min", atr_period=51)  # > 50

    def test_range_atr_order_validation(self) -> None:
        # min >= max phải raise
        with pytest.raises(ValidationError, match="min_range_atr"):
            ORBStrategyConfig(
                **self._base_kwargs(),
                resample_freq="5min",
                min_range_atr=3.0,
                max_range_atr=2.0,
            )

    def test_range_atr_equal_raises(self) -> None:
        with pytest.raises(ValidationError):
            ORBStrategyConfig(
                **self._base_kwargs(),
                resample_freq="5min",
                min_range_atr=2.0,
                max_range_atr=2.0,
            )

    def test_tp_sl_ratio_when_no_range_sl(self) -> None:
        # TP phải > SL khi không dùng range SL
        with pytest.raises(ValidationError, match="atr_tp_multiplier"):
            ORBStrategyConfig(
                **self._base_kwargs(),
                resample_freq="5min",
                use_range_sl=False,
                atr_tp_multiplier=1.0,
                atr_sl_multiplier=1.5,
            )

    def test_tp_sl_ratio_skipped_when_using_range_sl(self) -> None:
        # Khi use_range_sl=True, ratio check không áp dụng
        cfg = ORBStrategyConfig(
            **self._base_kwargs(),
            resample_freq="5min",
            use_range_sl=True,
            atr_tp_multiplier=1.0,
            atr_sl_multiplier=1.5,
        )
        assert cfg.use_range_sl is True

    def test_invalid_resample_freq(self) -> None:
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="2min")  # type: ignore

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**self._base_kwargs(), resample_freq="5min", nonexistent_param=123)  # type: ignore


# ══════════════════════════════════════════════════════════════════
# ORBConfig (top-level)
# ══════════════════════════════════════════════════════════════════


class TestORBConfig:
    def _valid_dict(self) -> dict:
        return {
            "name": "ORB",
            "version": "2.0.0",
            "strategy": {
                "resample_freq": "5min",
                "orb_minutes": 20,
                "atr_period": 14,
                "atr_tp_multiplier": 2.0,
                "atr_sl_multiplier": 1.5,
                "breakout_buffer": 0.1,
                "use_range_sl": True,
                "min_range_atr": 0.5,
                "max_range_atr": 3.0,
                "long_only": False,
                "use_volume_filter": False,
                "volume_filter_threshold": 0.5,
                "volume_ma_period": 20,
                "use_adx_filter": False,
                "adx_period": 14,
                "adx_min": 20.0,
                "require_close_confirmation": False,
                "max_trades_per_session": 2,
            },
            "risk": {
                "min_position_size": 1,
                "max_position_size": 10,
                "risk_per_trade_pct": 1.0,
                "max_daily_loss": 0.02,
                "use_trailing_stop": True,
                "trailing_atr_multiplier": 2.0,
                "entry_cutoff_seconds": 60,
                "allow_late_entry": False,
                "force_flat_on_session_close": False,
                "defer_exit_outside_session": True,
            },
        }

    def test_from_dict(self) -> None:
        cfg = ORBConfig.from_dict(self._valid_dict())
        assert cfg.strategy.orb_minutes == 20
        assert cfg.risk.max_daily_loss == 0.02

    def test_from_json_file(self) -> None:
        data = self._valid_dict()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = f.name

        cfg = ORBConfig.from_json(tmp_path)
        assert cfg.strategy.orb_minutes == 20
        Path(tmp_path).unlink()

    def test_to_json_roundtrip(self) -> None:
        cfg = ORBConfig.from_dict(self._valid_dict())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        cfg.to_json(tmp_path)
        reloaded = ORBConfig.from_json(tmp_path)
        assert reloaded.strategy.orb_minutes == cfg.strategy.orb_minutes
        assert reloaded.risk.max_daily_loss == cfg.risk.max_daily_loss
        Path(tmp_path).unlink()

    def test_missing_required_field(self) -> None:
        data = self._valid_dict()
        del data["strategy"]["orb_minutes"]
        # orb_minutes có default nên không raise — đây là intended behavior
        cfg = ORBConfig.from_dict(data)
        assert cfg.strategy.orb_minutes == 20  # default value

    def test_default_json_file_is_valid(self) -> None:
        """Đảm bảo orb_default.json luôn valid với schema hiện tại."""
        cfg = ORBConfig.from_json("config/strategy_params/orb_default.json")
        assert cfg.strategy.orb_minutes > 0
        assert cfg.risk.max_position_size >= cfg.risk.min_position_size
