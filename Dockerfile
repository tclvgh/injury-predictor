# Base image: Python 3.13 slim
FROM python:3.13-slim

# Install system dependencies required for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \  # Basic build tools \
        libpq-dev \        # PostgreSQL support \
        git \              # Version control \
        && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY .. .

# Expose the port Voila will run on
EXPOSE 8866

# Run Voila on the main notebook
CMD ["voila", "--Voila.ip=0.0.0.0", "--no-browser", "baseball.ipynb"]
