# Use official Python 3.13 slim image
FROM python:3.13-slim


# Prevent python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


# Install minimal system dependencies required to build some Python packages
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
build-essential \
git \
&& rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
&& pip install --no-cache-dir -r /app/requirements.txt


# Copy application code
COPY . /app


# Create a non-root user to run the app
RUN useradd --create-home appuser \
&& chown -R appuser:appuser /app
USER appuser


# Expose Flask port
EXPOSE 5000


# Default command: run the main script (which trains and starts the API per README)
CMD ["python", "main.py"]