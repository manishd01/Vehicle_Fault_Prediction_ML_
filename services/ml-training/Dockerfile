
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

ENV PYTHONPATH=/app

CMD ["bash"]



# FROM python:3.11-slim

# WORKDIR /app

# # System dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy Python dependencies
# COPY requirements.txt .

# # Install common ML dependencies
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Install CPU-only PyTorch
# RUN pip install --no-cache-dir \
#     torch \
#     --index-url https://download.pytorch.org/whl/cpu

# # Install TensorFlow
# RUN pip install --no-cache-dir tensorflow

# # Copy project files
# COPY . .

# ENV PYTHONPATH=/app

# CMD ["bash"]