"""
Test suite for secrets configuration.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.secrets import AppSecrets, BrokerSecrets, DBSecrets, RedisSecrets, get_secrets


class TestDBSecrets:
    """Test DBSecrets configuration."""

    def test_default_values(self):
        """Test DBSecrets default values."""
        db = DBSecrets()
        assert db.host == "localhost"
        assert db.port == 5432
        assert db.user == "postgres"
        assert db.name == "trading"

    def test_custom_values(self):
        """Test DBSecrets with custom values."""
        db = DBSecrets(
            host="db.example.com",
            port=5433,
            user="admin",
            password="secret123",
            name="prod_db",
        )
        assert db.host == "db.example.com"
        assert db.port == 5433
        assert db.user == "admin"
        assert db.name == "prod_db"

    def test_password_is_secret(self):
        """Test password is stored as SecretStr."""
        db = DBSecrets(password="secret123")
        assert str(db.password) != "secret123"  # SecretStr hides value
        assert db.password.get_secret_value() == "secret123"

    def test_to_psycopg2_kwargs(self):
        """Test conversion to psycopg2 connection kwargs."""
        db = DBSecrets(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            name="testdb",
        )
        kwargs = db.to_psycopg2_kwargs()
        assert kwargs == {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpass",
            "dbname": "testdb",
        }

    def test_port_validation(self):
        """Test port number validation."""
        with pytest.raises(ValidationError):
            DBSecrets(port=0)  # < 1

        with pytest.raises(ValidationError):
            DBSecrets(port=70000)  # > 65535


class TestRedisSecrets:
    """Test RedisSecrets configuration."""

    def test_default_values(self):
        """Test RedisSecrets default values."""
        redis = RedisSecrets()
        assert redis.host == "localhost"
        assert redis.port == 6379
        assert redis.password is None

    def test_custom_values(self):
        """Test RedisSecrets with custom values."""
        redis = RedisSecrets(
            host="redis.example.com",
            port=6380,
            password="redispass",
        )
        assert redis.host == "redis.example.com"
        assert redis.port == 6380

    def test_to_redis_kwargs_without_password(self):
        """Test conversion to redis kwargs without password."""
        redis = RedisSecrets(host="localhost", port=6379)
        kwargs = redis.to_redis_kwargs()
        assert kwargs == {"host": "localhost", "port": 6379}
        assert "password" not in kwargs

    def test_to_redis_kwargs_with_password(self):
        """Test conversion to redis kwargs with password."""
        redis = RedisSecrets(host="localhost", port=6379, password="secret")
        kwargs = redis.to_redis_kwargs()
        assert kwargs == {"host": "localhost", "port": 6379, "password": "secret"}


class TestBrokerSecrets:
    """Test BrokerSecrets configuration."""

    def test_default_values(self):
        """Test BrokerSecrets default values."""
        broker = BrokerSecrets()
        assert broker.username == ""
        assert broker.rest_base_url == "http://localhost:9090"
        assert broker.account_id_d1 == "D1"

    def test_custom_values(self):
        """Test BrokerSecrets with custom values."""
        broker = BrokerSecrets(
            username="trader1",
            password="brokerpass",
            rest_base_url="https://api.broker.com",
            account_id_d1="ACCOUNT123",
        )
        assert broker.username == "trader1"
        assert broker.rest_base_url == "https://api.broker.com"
        assert broker.account_id_d1 == "ACCOUNT123"


class TestAppSecrets:
    """Test AppSecrets top-level configuration."""

    def test_default_initialization(self):
        """Test AppSecrets initializes with default sub-configs."""
        app = AppSecrets()
        assert isinstance(app.db, DBSecrets)
        assert isinstance(app.redis, RedisSecrets)
        assert isinstance(app.broker, BrokerSecrets)

    def test_custom_sub_configs(self):
        """Test AppSecrets with custom sub-configs."""
        db = DBSecrets(host="custom-db")
        redis = RedisSecrets(host="custom-redis")
        broker = BrokerSecrets(username="custom-user")

        app = AppSecrets(db=db, redis=redis, broker=broker)
        assert app.db.host == "custom-db"
        assert app.redis.host == "custom-redis"
        assert app.broker.username == "custom-user"

    @patch.dict(
        "os.environ",
        {
            "DB_HOST": "env-db-host",
            "DB_PORT": "5433",
            "MARKET_REDIS_HOST": "env-redis-host",
        },
    )
    def test_loads_from_environment(self):
        """Test AppSecrets loads from environment variables."""
        app = AppSecrets()
        assert app.db.host == "env-db-host"
        assert app.db.port == 5433
        assert app.redis.host == "env-redis-host"


class TestGetSecrets:
    """Test get_secrets singleton function."""

    def test_returns_app_secrets(self):
        """Test get_secrets returns AppSecrets instance."""
        get_secrets.cache_clear()
        secrets = get_secrets()
        assert isinstance(secrets, AppSecrets)

    def test_singleton_behavior(self):
        """Test get_secrets returns same instance."""
        get_secrets.cache_clear()
        secrets1 = get_secrets()
        secrets2 = get_secrets()
        assert secrets1 is secrets2
