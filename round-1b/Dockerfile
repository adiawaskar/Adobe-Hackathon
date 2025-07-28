# File: round1b/Dockerfile
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies (minimal for PDF processing and building wheels)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy model and app code
COPY local_model /app/local_model
COPY app.py .

# Disable Chroma telemetry
ENV ANONYMIZED_TELEMETRY=False

# Run the app
CMD ["python", "app.py"]
