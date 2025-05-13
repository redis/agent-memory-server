import logging
import sys

import structlog


_configured = False


def configure_logging():
    """Configure structured logging for the application"""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.handlers = []  # Clear existing handlers
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)  # Set the desired logging level

    # logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stderr)

    # structlog.configure(
    #     processors=[
    #         structlog.processors.TimeStamper(fmt="iso"),
    #         structlog.stdlib.add_log_level,
    #         structlog.processors.JSONRenderer(),
    #     ],
    #     wrapper_class=structlog.stdlib.BoundLogger,
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    # )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Optional name for the logger (usually __name__)

    Returns:
        A configured logger instance
    """
    return structlog.get_logger(name)
