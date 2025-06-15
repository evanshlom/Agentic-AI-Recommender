FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user if it doesn't exist (devcontainer image already has it)
# This ensures compatibility with different base images
RUN if ! id -u vscode > /dev/null 2>&1; then \
    useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; \
    fi

# Set working directory
WORKDIR /workspace

# Fix numpy compatibility issue first
RUN pip install --upgrade pip && \
    pip install "numpy<2.0"

# Install PyTorch CPU version (to avoid GPU dependencies in container)
RUN pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Switch to vscode user
USER vscode