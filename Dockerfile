# Image Recognition Training Container
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trainer package
COPY trainer/ ./trainer/

# Set Python path
ENV PYTHONPATH=/app

# Define entrypoint
ENTRYPOINT ["python", "-m", "trainer.cli"]

# Default command (shows help)
CMD ["--help"]
