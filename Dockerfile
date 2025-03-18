FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY redis_memory_server ./redis_memory_server

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Run the API server
CMD ["python", "-m", "redis_memory_server.main"]
