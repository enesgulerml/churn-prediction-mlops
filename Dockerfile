# --- STAGE 1: BUILDER (Construction Site) ---
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

# --- STAGE 2: RUNTIME (Production Ready & Secure) ---
# Use a fresh slim image for the final container
FROM python:3.10-slim as runtime

# 1. SECURITY: Create a non-root group and user
RUN groupadd -r mlops && useradd -r -g mlops mlops_user

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

# 2. SECURITY: Change ownership of the application directory to the non-root user
RUN chown -R mlops_user:mlops /app

# 3. SECURITY: Switch to the non-root user for execution
USER mlops_user

# Default command (will be overridden by docker-compose)
CMD ["python", "src/app.py"]