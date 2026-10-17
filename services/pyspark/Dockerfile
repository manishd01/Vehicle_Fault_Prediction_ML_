
#  firrst time: docker build -f Dockerfile.java-base -t vehicle-fault-python-java:3.11 .
# or docker build -f .\services\pyspark\Dockerfile.java-base -t vehicle-fault-python-java:3.11 .\services\pyspark
# after : docker compose build pyspark


# resuming: python .\run_pipeline.py --resume 20261017T173635679484Z
FROM vehicle-fault-python-java:3.11

WORKDIR /app

# Install Python dependencies separately from application code.
COPY requirements.txt .

# Keep downloaded package files in a persistent BuildKit cache.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code last.
COPY . .

ENV PYTHONPATH=/app

CMD ["bash"]
