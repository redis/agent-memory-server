FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Node.js 22 directly (matching working devcontainer)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-tools \
    jq \
    vim \
    nano \
    htop \
    ncdu \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager) for root
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set up npm global directory for vscode user (like working devcontainer)
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R vscode:vscode /usr/local/share/npm-global

# Configure npm to use user-writable global directory
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="$PATH:/usr/local/share/npm-global/bin"

# Set DEVCONTAINER environment variable (important for Claude)
ENV DEVCONTAINER=true

# Create workspace and config directories with proper ownership
RUN mkdir -p /workspace /home/vscode/.claude && \
    chown -R vscode:vscode /workspace /home/vscode/.claude

WORKDIR /workspace

# Switch to vscode user for remaining setup
USER vscode

# Install uv for vscode user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.cargo/bin:$PATH"

# Install Claude Code CLI globally as vscode user
RUN npm install -g @anthropic-ai/claude-code@latest && \
    # Create a shell wrapper that explicitly calls /usr/bin/node (bypasses nvm/PATH issues)
    rm -f /usr/local/share/npm-global/bin/claude && \
    echo '#!/bin/bash' > /usr/local/share/npm-global/bin/claude && \
    echo 'exec /usr/bin/node /usr/local/share/npm-global/lib/node_modules/@anthropic-ai/claude-code/cli.js "$@"' >> /usr/local/share/npm-global/bin/claude && \
    chmod +x /usr/local/share/npm-global/bin/claude

# Set default shell and editor
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Add useful aliases
RUN echo 'alias yolo="claude --dangerously-skip-permissions"' >> /home/vscode/.zshrc && \
    echo 'alias yolo="claude --dangerously-skip-permissions"' >> /home/vscode/.bashrc
