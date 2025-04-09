import logging
import sys

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    # Suppress overly verbose logs from libraries if needed
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)