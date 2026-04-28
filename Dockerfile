# --- System creation Layer ---
FROM alpine/curl AS vscode-installer
RUN mkdir /aichor
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz
RUN tar -xf /aichor/vscode_cli.tar.gz -C /aichor

FROM python:3.10-slim AS core

# System deps. egl/gles libs needed for MuJoCo headless rendering.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    sudo \
    vim \
    nano \
    curl \
    wget \
    unzip \
    tar \
    zip \
    gzip \
    bzip2 \
    procps \
    libegl1 \
    libgl1 \
    libglu1-mesa \
    libosmesa6 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Non-root user (matches Mava cluster convention).
RUN useradd -m -u 1000 -s /bin/bash app

# Use the system-wide Python (no venv inside container).
ENV UV_SYSTEM_PYTHON=1
WORKDIR /home/app/jaxgcrl

# --- Dependency Installation Layer ---
# Copy lockfile + project metadata first for caching.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv,id=project-3aa787a5-8b75-43c3-a5f1-976597803e01 \
    uv sync --locked --no-install-project

# --- Application Code Layer ---
COPY . .

# Install local project + post-sync fix for CUDA libs that uv may strip.
RUN --mount=type=cache,target=/root/.cache/uv,id=project-3aa787a5-8b75-43c3-a5f1-976597803e01 \
    uv sync --locked && \
    uv pip install --force-reinstall --no-deps nvidia-cudnn-cu12==8.9.7.29

# Headless MuJoCo by default.
ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Optional: VS Code remote (mirrors Mava setup).
COPY --from=vscode-installer /aichor /aichor
RUN uv pip install nvitop

# Hand ownership to non-root user so runtime writes (ckpts, logs) work without sudo.
RUN chown -R app:app /home/app
USER app

# Tensorboard / other UIs.
EXPOSE 6006
