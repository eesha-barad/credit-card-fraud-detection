FROM python:3.11-slim

WORKDIR /app

# System deps needed by matplotlib/scikit-learn wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY examples/ examples/
COPY dataset/README.md dataset/README.md

# Default: run the quick example against the bundled synthetic sample data.
# For the real dataset, mount it and pass --config pointing at it, e.g.:
#   docker run -v $(pwd)/dataset:/app/dataset fraud-detection python -m src.pipeline
ENTRYPOINT ["python"]
CMD ["examples/run_example.py"]
