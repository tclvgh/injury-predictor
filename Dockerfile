# Base image: Python 3.13 slim
FROM python:3.13-slim

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
