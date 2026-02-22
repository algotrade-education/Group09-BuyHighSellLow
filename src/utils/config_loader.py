import json
import logging
from pathlib import Path
from typing import Any, Dict


def load_config(
    config_path: str = "config/strategy_params/default.json",
) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)

    # Check if path exists directly
    if not path.exists():
        # Try finding it relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / config_path

    if not path.exists():
        logging.warning(f"Config file {config_path} not found. Using empty defaults.")
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file {path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error loading config {path}: {e}")
        return {}
