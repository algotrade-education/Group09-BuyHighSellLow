"""Unit tests for broker credentials management in config.secrets.

Tests cover:
- BrokerCredentials dataclass
- get_broker_credentials() resolution strategies
- Error handling for missing credentials
"""

from unittest.mock import Mock, patch

import pytest

from config.secrets import BrokerCredentials, get_broker_credentials

# --- BrokerCredentials Tests ---


def test_broker_credentials_dataclass():
    """Test BrokerCredentials dataclass creation."""
    creds = BrokerCredentials(
        rest_base_url="https://api.broker.com",
        username="testuser",
        password="testpass",
        sender_comp_id="SENDER123",
        target_comp_id="TARGET456",
        socket_connect_host="broker.com",
        socket_connect_port=5001,
        default_sub_account="D1",
    )

    assert creds.rest_base_url == "https://api.broker.com"
    assert creds.username == "testuser"
    assert creds.password == "testpass"
    assert creds.sender_comp_id == "SENDER123"
    assert creds.target_comp_id == "TARGET456"
    assert creds.socket_connect_host == "broker.com"
    assert creds.socket_connect_port == 5001
    assert creds.default_sub_account == "D1"


def test_broker_credentials_defaults():
    """Test BrokerCredentials default values."""
    creds = BrokerCredentials(
        rest_base_url="https://api.broker.com",
        username="testuser",
        password="testpass",
        sender_comp_id="SENDER123",
    )

    assert creds.target_comp_id == "SERVER"
    assert creds.socket_connect_host == "localhost"
    assert creds.socket_connect_port == 5001
    assert creds.default_sub_account == "D1"


def test_broker_credentials_to_client_kwargs():
    """Test to_client_kwargs() method."""
    creds = BrokerCredentials(
        rest_base_url="https://api.broker.com",
        username="testuser",
        password="testpass",
        sender_comp_id="SENDER123",
    )

    kwargs = creds.to_client_kwargs()

    assert kwargs["rest_base_url"] == "https://api.broker.com"
    assert kwargs["username"] == "testuser"
    assert kwargs["password"] == "testpass"
    assert kwargs["sender_comp_id"] == "SENDER123"
    assert kwargs["target_comp_id"] == "SERVER"
    assert kwargs["socket_connect_host"] == "localhost"
    assert kwargs["socket_connect_port"] == 5001
    assert kwargs["default_sub_account"] == "D1"


# --- get_broker_credentials() Tests ---


@patch("config.secrets.get_secrets")
def test_get_broker_credentials_explicit_sender_id(mock_get_secrets):
    """Test resolution with explicit SenderCompID parameter."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    creds = get_broker_credentials(sender_comp_id="EXPLICIT_ID", enable_api_resolution=False)

    assert creds.sender_comp_id == "EXPLICIT_ID"
    assert creds.username == "testuser"


@patch("config.secrets.get_secrets")
def test_get_broker_credentials_from_env(mock_get_secrets):
    """Test resolution from environment variable."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.sender_comp_id = "ENV_ID"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    creds = get_broker_credentials(enable_api_resolution=False)

    assert creds.sender_comp_id == "ENV_ID"


@patch("config.secrets.get_secrets")
@patch("config.secrets._resolve_sender_comp_id_from_api")
def test_get_broker_credentials_from_api(mock_api_resolve, mock_get_secrets):
    """Test resolution from REST API."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    # Mock API resolution
    mock_api_resolve.return_value = "API_ID"

    creds = get_broker_credentials(enable_api_resolution=True)

    assert creds.sender_comp_id == "API_ID"
    mock_api_resolve.assert_called_once_with("https://api.broker.com", "testuser", "testpass")


@patch("config.secrets.get_secrets")
@patch("config.secrets._resolve_sender_comp_id_from_api")
def test_get_broker_credentials_api_fallback_to_env(mock_api_resolve, mock_get_secrets):
    """Test fallback to env when API fails."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.sender_comp_id = "ENV_FALLBACK"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    # Mock failed API resolution
    mock_api_resolve.return_value = None

    creds = get_broker_credentials(enable_api_resolution=True)

    assert creds.sender_comp_id == "ENV_FALLBACK"


@patch("config.secrets.get_secrets")
def test_get_broker_credentials_no_resolution_raises(mock_get_secrets):
    """Test that missing SenderCompID raises ValueError."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.sender_comp_id = ""  # Empty sender_comp_id
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    with pytest.raises(ValueError, match="Could not resolve SenderCompID"):
        get_broker_credentials(enable_api_resolution=False)


@patch("config.secrets.get_secrets")
def test_get_broker_credentials_custom_socket_config(mock_get_secrets):
    """Test custom socket configuration from environment."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "main"
    mock_secrets.fix.sender_comp_id = "SENDER"
    mock_secrets.fix.target_comp_id = "CUSTOM_TARGET"
    mock_secrets.fix.socket_connect_host = "custom.broker.com"
    mock_secrets.fix.socket_connect_port = 9999
    mock_get_secrets.return_value = mock_secrets

    creds = get_broker_credentials(enable_api_resolution=False)

    assert creds.target_comp_id == "CUSTOM_TARGET"
    assert creds.socket_connect_host == "custom.broker.com"
    assert creds.socket_connect_port == 9999


# --- Resolution Priority Tests ---


@patch("config.secrets.get_secrets")
@patch("config.secrets._resolve_sender_comp_id_from_api")
def test_resolution_priority_explicit_over_api(mock_api_resolve, mock_get_secrets):
    """Test that explicit SenderCompID takes priority over API."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    # Mock API resolution (should not be called)
    mock_api_resolve.return_value = "API_ID"

    creds = get_broker_credentials(sender_comp_id="EXPLICIT_ID", enable_api_resolution=True)

    # Explicit should win
    assert creds.sender_comp_id == "EXPLICIT_ID"

    # API should not be called
    mock_api_resolve.assert_not_called()


@patch("config.secrets.get_secrets")
@patch("config.secrets._resolve_sender_comp_id_from_api")
def test_resolution_priority_api_over_env(mock_api_resolve, mock_get_secrets):
    """Test that API resolution takes priority over environment variable."""
    # Mock secrets
    mock_secrets = Mock()
    mock_secrets.broker.rest_base_url = "https://api.broker.com"
    mock_secrets.broker.username = "testuser"
    mock_secrets.broker.password.get_secret_value.return_value = "testpass"
    mock_secrets.broker.account_id_d1 = "D1"
    mock_secrets.fix.sender_comp_id = "ENV_ID"
    mock_secrets.fix.target_comp_id = "SERVER"
    mock_secrets.fix.socket_connect_host = "localhost"
    mock_secrets.fix.socket_connect_port = 5001
    mock_get_secrets.return_value = mock_secrets

    # Mock API resolution
    mock_api_resolve.return_value = "API_ID"

    creds = get_broker_credentials(enable_api_resolution=True)

    # API should win over env
    assert creds.sender_comp_id == "API_ID"
