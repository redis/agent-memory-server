#!/usr/bin/env python3
"""Run the Redis Agent Memory Server."""

import os

import uvicorn

from redis_memory_server.main import on_start_logger


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    on_start_logger(port)
    uvicorn.run("redis_memory_server.main:app", host="0.0.0.0", port=port, reload=True)
