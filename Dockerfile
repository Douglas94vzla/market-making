# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (build-essential for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install the local package in editable mode (as per requirements.txt)
RUN pip install -e .

# Set environment variables (these can be overridden by .env or docker-compose)
ENV PYTHONUNBUFFERED=1

# Run the application
# Default to BTCUSDT indefinitely (duration 0)
CMD ["python", "src/run.py", "--symbol", "BTCUSDT", "--duration", "0"]
