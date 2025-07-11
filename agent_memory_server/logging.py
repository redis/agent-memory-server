import logging
import sys

import structlog

from agent_memory_server.config import settings


_configured = False


def configure_logging():
    """Configure structured logging for the application"""
    global _configured
    if _configured:
        return

    # Configure standard library logging based on settings.log_level
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)  # Use stderr instead of stdout
    handler.setLevel(level)
    logging.basicConfig(level=level, handlers=[handler], format="%(message)s")

    # Configure structlog with processors honoring the log level and structured output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def configure_mcp_logging():
    """Configure logging specifically for MCP server in stdio mode"""
    global _configured

    # Clear any existing handlers and configuration
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure stderr-only logging for MCP stdio mode
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root_logger.addHandler(stderr_handler)
    root_logger.setLevel(level)

    # Configure structlog to also use stderr
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
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
