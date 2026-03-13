# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /app/models

# Copy project files
COPY flaskapp.py .
COPY functionality.py .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "flaskapp.py"]