FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Copy the rest of the application
COPY . .

# Set default port (or use the one passed via environment)
ENV PORT=8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:server"]