import logging
import logging.config
from pathlib import Path
from config.settings import LOG_LEVEL, LOG_FILE

def setup_logging():
    """Configure centralized logging for the application."""
    log_file = Path(LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'file': {
                'level': LOG_LEVEL,
                'class': 'logging.FileHandler',
                'filename': LOG_FILE,
                'formatter': 'standard'
            },
            'console': {
                'level': LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'loggers': {
            '': {
                'handlers': ['file', 'console'],
                'level': LOG_LEVEL,
                'propagate': True
            },
        }
    })
