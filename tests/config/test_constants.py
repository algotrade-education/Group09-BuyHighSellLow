"""
Test suite for config constants.
"""

from config.constants import (
    CACHE_DIR,
    DATETIME_COLUMN,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SYMBOL,
    VN30F_COMMISSION_PER_CONTRACT,
    VN30F_CONTRACT_MULTIPLIER,
    VN30F_MARGIN_PER_CONTRACT,
)


class TestConstants:
    """Test config constants are defined correctly."""

    def test_default_initial_capital(self):
        """Test DEFAULT_INITIAL_CAPITAL is defined."""
        assert isinstance(DEFAULT_INITIAL_CAPITAL, float)
        assert DEFAULT_INITIAL_CAPITAL == 500_000_000.00

    def test_vn30f_contract_multiplier(self):
        """Test VN30F_CONTRACT_MULTIPLIER is defined."""
        assert isinstance(VN30F_CONTRACT_MULTIPLIER, float)
        assert VN30F_CONTRACT_MULTIPLIER == 100_000.00

    def test_vn30f_commission_per_contract(self):
        """Test VN30F_COMMISSION_PER_CONTRACT is defined."""
        assert isinstance(VN30F_COMMISSION_PER_CONTRACT, float)
        assert VN30F_COMMISSION_PER_CONTRACT == 4_750.00

    def test_vn30f_margin_per_contract(self):
        """Test VN30F_MARGIN_PER_CONTRACT is defined."""
        assert isinstance(VN30F_MARGIN_PER_CONTRACT, float)
        assert VN30F_MARGIN_PER_CONTRACT == 9_000_000.00

    def test_default_symbol(self):
        """Test DEFAULT_SYMBOL is defined."""
        assert isinstance(DEFAULT_SYMBOL, str)
        assert DEFAULT_SYMBOL == "VN30F1M"

    def test_datetime_column(self):
        """Test DATETIME_COLUMN is defined."""
        assert isinstance(DATETIME_COLUMN, str)
        assert DATETIME_COLUMN == "datetime"

    def test_cache_dir(self):
        """Test CACHE_DIR is defined."""
        assert isinstance(CACHE_DIR, str)
        assert CACHE_DIR == "data/cache/"

    def test_all_constants_are_immutable_types(self):
        """Test all constants use immutable types."""
        constants = [
            DEFAULT_INITIAL_CAPITAL,
            VN30F_CONTRACT_MULTIPLIER,
            VN30F_COMMISSION_PER_CONTRACT,
            VN30F_MARGIN_PER_CONTRACT,
            DEFAULT_SYMBOL,
            DATETIME_COLUMN,
            CACHE_DIR,
        ]
        for const in constants:
            assert isinstance(const, (str, int, float, bool, tuple))
