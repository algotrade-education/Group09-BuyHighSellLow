"""
Credentials for DB and external services
Uses pydantic.SecretStr to ensure that sensitive information is not accidentally exposed in logs or error messages.

Load automatically from environment variables or .env file using pydantic's BaseSettings.

Usage:
    from config.secrets import get_secrets
    secrets = get_secrets()
    conn = psycopg2.connect(
        **secrets.db.to_psycopg2_kwargs()
    )

    # For broker credentials with SenderCompID resolution:
    from config.secrets import get_broker_credentials
    creds = get_broker_credentials()
    client = PaperBrokerClient(**creds.to_client_kwargs())
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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


# --- Broker Credentials Manager ---


@dataclass
class BrokerCredentials:
    """Complete broker credentials with resolved SenderCompID.

    Attributes:
        rest_base_url: Base URL for broker REST API.
        username: Broker account username.
        password: Broker account password (plain text).
        sender_comp_id: FIX protocol sender company ID (resolved).
        target_comp_id: FIX protocol target company ID.
        socket_host: FIX socket connection host.
        socket_port: FIX socket connection port.
    """

    rest_base_url: str
    username: str
    password: str
    sender_comp_id: str
    target_comp_id: str = "SERVER"
    socket_host: str = "localhost"
    socket_port: int = 5001

    def to_client_kwargs(self) -> dict:
        """Convert to kwargs dict for PaperBrokerClient constructor."""
        return {
            "rest_base_url": self.rest_base_url,
            "username": self.username,
            "password": self.password,
            "sender_comp_id": self.sender_comp_id,
            "target_comp_id": self.target_comp_id,
            "socket_host": self.socket_host,
            "socket_port": self.socket_port,
        }


def _resolve_sender_comp_id_from_api(
    rest_base_url: str,
    username: str,
    password: str,
) -> str | None:
    """Attempt to resolve SenderCompID from broker REST API.

    Args:
        rest_base_url: Broker REST API base URL.
        username: Broker username.
        password: Broker password.

    Returns:
        SenderCompID string if successful, None otherwise.
    """
    try:
        import httpx

        logger.debug("Attempting to resolve SenderCompID from REST API at %s", rest_base_url)

        # Construct auth endpoint (adjust based on actual API)
        auth_endpoint = f"{rest_base_url}/api/auth/sender-comp-id"

        response = httpx.get(
            auth_endpoint,
            auth=(username, password),
            timeout=5.0,
        )

        if response.status_code == 200:
            data = response.json()
            sender_comp_id: str = data.get("sender_comp_id")

            if sender_comp_id:
                logger.info("Resolved SenderCompID from REST API: %s", sender_comp_id)
                return sender_comp_id

        logger.debug("REST API did not return SenderCompID (status=%d)", response.status_code)
        return None

    except ImportError:
        logger.debug("httpx not available, skipping API resolution")
        return None
    except Exception as exc:
        logger.debug("Failed to resolve SenderCompID from REST API: %s", exc, exc_info=False)
        return None


def get_broker_credentials(
    sender_comp_id: str | None = None,
    enable_api_resolution: bool = True,
) -> BrokerCredentials:
    """Get complete broker credentials with resolved SenderCompID.

    Resolution order for SenderCompID:
    1. Explicit sender_comp_id parameter
    2. REST API query (if enable_api_resolution=True)
    3. Environment variable SENDER_COMP_ID
    4. Raise ValueError if none available

    Args:
        sender_comp_id: Explicit SenderCompID (optional, will be resolved if not provided).
        enable_api_resolution: Whether to try resolving SenderCompID from REST API.

    Returns:
        BrokerCredentials instance with all fields populated.

    Raises:
        ValueError: If SenderCompID cannot be resolved.

    Example:
        >>> creds = get_broker_credentials()
        >>> client = PaperBrokerClient(**creds.to_client_kwargs())
    """
    secrets = get_secrets()

    # Get base credentials from secrets
    rest_base_url = secrets.broker.rest_base_url
    username = secrets.broker.username
    password = secrets.broker.password.get_secret_value()

    # Get optional FIX connection details from environment
    target_comp_id = os.getenv("TARGET_COMP_ID", "SERVER")
    socket_host = os.getenv("SOCKET_CONNECT_HOST", "localhost")
    socket_port = int(os.getenv("SOCKET_CONNECT_PORT", "5001"))

    # Resolve SenderCompID using multiple strategies
    resolved_sender_id = None

    # Strategy 1: Explicit parameter
    if sender_comp_id:
        logger.info("Using explicit SenderCompID: %s", sender_comp_id)
        resolved_sender_id = sender_comp_id

    # Strategy 2: REST API resolution
    elif enable_api_resolution:
        resolved_sender_id = _resolve_sender_comp_id_from_api(rest_base_url, username, password)

    # Strategy 3: Environment variable
    if not resolved_sender_id:
        env_sender_id = os.getenv("SENDER_COMP_ID")
        if env_sender_id:
            logger.info("Using SenderCompID from environment variable: %s", env_sender_id)
            resolved_sender_id = env_sender_id

    # No resolution strategy succeeded
    if not resolved_sender_id:
        raise ValueError(
            "Could not resolve SenderCompID. Please provide it via:\n"
            "  1. Parameter: get_broker_credentials(sender_comp_id='YOUR_ID')\n"
            "  2. Environment variable: SENDER_COMP_ID=YOUR_ID\n"
            "  3. Ensure REST API is accessible for automatic resolution"
        )

    return BrokerCredentials(
        rest_base_url=rest_base_url,
        username=username,
        password=password,
        sender_comp_id=resolved_sender_id,
        target_comp_id=target_comp_id,
        socket_host=socket_host,
        socket_port=socket_port,
    )
