FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    sentencepiece \
    sacrebleu

# Copy application files
COPY input.txt .
COPY reference.txt .
COPY translate.py .
COPY evaluate.py .

# Default command: run translation and then evaluation
CMD ["sh", "-c", "python translate.py && python evaluate.py"]
