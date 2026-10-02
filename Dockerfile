FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by XGBoost (libgomp1) and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for efficient Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and MLflow runs directory (if pre-trained models are included)
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to monitor Streamlit container status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit with non-blocking binding
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]