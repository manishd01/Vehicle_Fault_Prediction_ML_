
# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Build the Python wheel
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m pip install --no-cache-dir --upgrade pip build setuptools wheel

COPY pyproject.toml .
COPY src/ ./src/

RUN python -m build --wheel


# ==========================================
# Stage 2: Runtime image
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Keep this layer independent of source-code changes
COPY requirements.txt .

# Cache downloaded Python packages persistently in BuildKit
RUN --mount=type=cache,id=vehicle-fault-pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Install CPU-only PyTorch using the same cache
RUN --mount=type=cache,id=vehicle-fault-pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the wheel built in Stage 1
COPY --from=builder /build/dist/ /tmp/wheels/

RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

ENV PYTHONPATH=/app

CMD ["bash"]
# --------------------------------------


# # syntax=docker/dockerfile:1

# FROM python:3.11-slim

# WORKDIR /app

# # Keep this layer independent of source-code changes
# COPY requirements.txt .

# # Cache downloaded Python packages persistently in BuildKit
# RUN --mount=type=cache,id=vehicle-fault-pip-cache,target=/root/.cache/pip,sharing=locked \
#     python -m pip install --upgrade pip && \
#     python -m pip install -r requirements.txt

# # Install CPU-only PyTorch using the same cache
# RUN --mount=type=cache,id=vehicle-fault-pip-cache,target=/root/.cache/pip,sharing=locked \
#     python -m pip install torch \
#     --index-url https://download.pytorch.org/whl/cpu

# # Source code changes should not invalidate dependency layers
# COPY . .

# ENV PYTHONPATH=/app

# CMD ["bash"]

