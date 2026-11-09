#!/bin/bash
set -e

# ============================================
# Standalone Agent Memory Server Entrypoint
# ============================================
# This script initializes the environment before
# starting supervisord with Redis, API, and worker.

# Create data directory if it doesn't exist
mkdir -p /data

# Create supervisor log directory
mkdir -p /var/log/supervisor

# Ensure proper permissions on data directory
# (Redis needs to write here)
chown -R redis:redis /data 2>/dev/null || true

echo "============================================"
echo "Agent Memory Server (Standalone)"
echo "============================================"
echo "Redis data directory: /data"
echo "API server: http://localhost:8000"
echo "============================================"

# Wait for any pre-start hooks (future use)
if [ -f /app/pre-start.sh ]; then
    echo "Running pre-start hook..."
    /app/pre-start.sh
fi

# Execute the main command (supervisord)
exec "$@"
