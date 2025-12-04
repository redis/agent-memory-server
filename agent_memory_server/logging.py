import logging
import sys

import structlog

from agent_memory_server.config import settings


_configured = False


def configure_logging():
    """Configure structured logging for the application.

    NOTE:
        We intentionally bind handlers to ``sys.__stderr__`` instead of ``sys.stderr``.
        Test runners like Click's ``CliRunner`` temporarily replace and then close
        ``sys.stderr`` to capture output. If we bind logging handlers to that
        temporary stream, any later log writes will raise ``ValueError: I/O
        operation on closed file`` once the runner closes it. Using
        ``sys.__stderr__`` ensures we always log to the original process stderr
        that remains open for the lifetime of the process.
    """
    global _configured
    if _configured:
        return

    # Configure standard library logging based on settings.log_level
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Use the original stderr stream so logging isn't tied to temporary test
    # runners' replacements of sys.stderr (which may be closed later).
    handler = logging.StreamHandler(sys.__stderr__)
    handler.setLevel(level)
    logging.basicConfig(level=level, handlers=[handler], format="%(message)s")

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("docket.worker").setLevel(logging.WARNING)
    logging.getLogger("agent_memory_server.dependencies").setLevel(logging.WARNING)

    # Set PyTorch to be less verbose about device selection
    logging.getLogger("torch").setLevel(logging.WARNING)

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

    # Configure stderr-only logging for MCP stdio mode. Use the original
    # stderr stream so MCP logging is not affected by temporary replacements
    # of sys.stderr (for example, in tests).
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    stderr_handler = logging.StreamHandler(sys.__stderr__)
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
