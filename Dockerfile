# Start from a lightweight base image with a specific Python version
FROM python:3.11-slim

# Install essential packages for pyenv and Python builds
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv for Python version management
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
RUN curl https://pyenv.run | bash

# Install pipx and use it to install poetry
ENV PIPX_BIN_DIR="/root/.local/bin"
ENV PATH="$PIPX_BIN_DIR:$PATH"
RUN pip install --no-cache-dir pipx && \
    pipx install poetry

# Install commonly used Python versions with pyenv for flexibility
RUN pyenv install 3.8.10 && pyenv install 3.9.13 && pyenv install 3.10.5 && pyenv install 3.11.6

# Set a default global Python version (can be overridden by individual projects)
RUN pyenv global 3.11.6

# Configure poetry to create virtual environments within each project directory
RUN poetry config virtualenvs.in-project true

# Set work directory (to be replaced in downstream images by project-specific directories)
WORKDIR /app

# Expose poetry and pyenv paths for downstream images
ENV PATH="$PYENV_ROOT/shims:$PIPX_BIN_DIR:$PATH"

# Optionally add an entrypoint to make poetry commands easier to run in derived images
ENTRYPOINT ["poetry"]
