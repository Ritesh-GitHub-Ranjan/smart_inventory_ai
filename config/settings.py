import os
from pathlib import Path

__version__ = "1.0.0"
BASE_DIR = Path(__file__).parent.parent

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/db/inventory.db")

# LLM Configuration
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434/api/generate")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "tinyllama")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", BASE_DIR / "logs" / "application.log")
