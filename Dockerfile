# --- STAGE 1: BUILDER ---
# Use a slim python image for the build process
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for building Python packages (gcc, libpq-dev)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: RUNTIME ---
# Use a fresh slim image for the final container
FROM python:3.10-slim as runtime

WORKDIR /app

# Set environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install only the necessary runtime system libraries
# libpq5 -> Required for PostgreSQL connection
# libgomp1 -> Required for XGBoost parallelism (Critical)
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-installed virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY . .

# Default command (will be overridden by docker-compose)
CMD ["python", "src/app.py"]