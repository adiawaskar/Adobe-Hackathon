# File: round1b/Dockerfile
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app
# File: round1b/Dockerfile
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install scikit-learn with minimal dependencies
RUN pip install --no-cache-dir scikit-learn==1.0.2

# Copy application code
COPY app.py .

CMD ["python", "app.py"]