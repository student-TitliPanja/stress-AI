# Use Python 3.11 slim as the baseline for a smaller footprint
FROM python:3.11-slim

# Prevent Python from writing .pyc files & force unbuffered stdout for logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install critical system-level dependencies for computer vision and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install the huge AI dependencies
# (We increase timeout because TensorFlow downloads can be slow)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the dynamic deployment port (default 5000)
ENV PORT=5000
EXPOSE $PORT

# Run the application (Production deployment command)
CMD ["python", "app.py"]
