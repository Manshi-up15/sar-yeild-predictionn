# Use official Python slim image for lightweight production deployment
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for geospatial packages (Rasterio, GDAL) and compiler utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source modules, configuration file, and directories
COPY src/ /app/src/
COPY .env.example /app/.env
COPY README.md /app/

# Expose FastAPI port (8000) and Streamlit port (8501)
EXPOSE 8000
EXPOSE 8501

# Default command can be overridden by docker-compose
CMD ["uvicorn", "crop_yield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
