FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV PORT=11434
ENV HOST=0.0.0.0
ENV LOG_LEVEL=info

# Expose port
EXPOSE 11434

# Run the application
CMD ["python", "run.py"]
