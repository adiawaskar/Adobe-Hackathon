# File: round1b/Dockerfile
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# Add scikit-learn explicitly (if not already in requirements.txt)
RUN pip install --no-cache-dir scikit-learn==1.0.2

# Copy model and app code
COPY local_model /app/local_model
COPY app.py .

ENV ANONYMIZED_TELEMETRY=False

# Set command
CMD ["python", "app.py"]
