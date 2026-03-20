
"""
logging_config.py — Logging Configuration
-------------------------------------------
Configures application-wide logging for the Document Intelligence Hub.

Sets up two handlers:
- FileHandler  — writes INFO-level logs to logs/app.log (persistent)
- StreamHandler — echoes logs to the terminal during development

Must be called as the very first step in app.py, before any other
module is imported or initialized, to ensure all log messages are captured.

Primary function: configure_logging()
Log file location: logs/app.log
"""

import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )