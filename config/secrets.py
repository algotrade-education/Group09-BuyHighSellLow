"""
Credentials for DB and external services
Uses pydantic.SecretStr to ensure that sensitive information is not accidentally exposed in logs or error messages.

Load automatically from environment variables or .env file using pydantic's BaseSettings.

Usage:
    from config.secrets import get_secrets
    secrets = get_secrets()
    conn = psycopg2.connect(
        host=secrets.db_host.get_secret_value(),
        port=secrets.db_port,
        user=secrets.db_user.get_secret_value(),
        password=secrets.db_password.get_secret_value(),
        dbname=secrets.db_name.get_secret_value()
    )
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSecrets(BaseSettings):
    """PostgreSQL database credentials and connection details."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",  # Environment variables should start with DB_
        env_file=".env",  # Load from .env file if present
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in the .env file
    )

    host: str = Field(default="localhost", description="Database host address")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port number")
    user: str = Field(default="postgres", description="Database username")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")
    name: str = Field(default="trading", description="Database name")

    def to_psycopg2_kwargs(self) -> dict:
        """Convert the secrets to a dictionary format suitable for psycopg2 connection."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password.get_secret_value(),
            "dbname": self.name,
        }


class RedisSecrets(BaseSettings):
    """Redis connection details."""

    model_config = SettingsConfigDict(
        env_prefix="MARKET_REDIS_",  # Environment variables should start with REDIS_
        env_file=".env",  # Load from .env file if present
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in the .env file
    )

    host: str = Field(default="localhost", description="Redis host address")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port number")
    password: SecretStr | None = Field(default=None, description="Redis password")

    def to_redis_kwargs(self) -> dict:
        kwargs: dict = {"host": self.host, "port": self.port}
        if self.password:
            kwargs["password"] = self.password.get_secret_value()
        return kwargs


class BrokerSecrets(BaseSettings):
    """Paper broker REST API credentials."""

    model_config = SettingsConfigDict(
        env_prefix="PAPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    username: str = Field(default="")
    password: SecretStr = Field(default=SecretStr(""))
    rest_base_url: str = Field(default="http://localhost:9090")
    account_id_d1: str = Field(default="D1")


class AppSecrets(BaseSettings):
    """Top-level secrets container."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db: DBSecrets = Field(default_factory=DBSecrets)
    redis: RedisSecrets = Field(default_factory=RedisSecrets)
    broker: BrokerSecrets = Field(default_factory=BrokerSecrets)


@lru_cache(maxsize=1)
def get_secrets() -> AppSecrets:
    """
    Lazy singleton - load secrets on first call and cache the result for subsequent calls.

    Raises:
        ValidationError: If the environment variables or .env file contain invalid values
        that do not conform to the expected types or constraints defined in the secrets schemas.
    """
    return AppSecrets()
