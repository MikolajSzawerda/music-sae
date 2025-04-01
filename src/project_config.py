from pathlib import Path

from dotenv import load_dotenv, dotenv_values
import logging.config

# Load environment variables from .env file if it exists
load_dotenv()


def setup_logging():
    """Configures logging from the logging.conf file."""
    LOGGING_CONFIG_PATH = PROJ_ROOT / "logging.conf"
    if LOGGING_CONFIG_PATH.exists():
        logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
    else:
        print(f"Warning: Logging configuration file not found at {LOGGING_CONFIG_PATH}")
        logging.basicConfig(level=logging.INFO)  # Fallback to basic logging


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

config = dotenv_values(PROJ_ROOT / ".env")
setup_logging()

logger = logging.getLogger(__name__)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
