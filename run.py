#!/usr/bin/env python3
"""Run the Redis Memory Server."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "redis_memory_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
