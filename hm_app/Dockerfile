# ==========================================
# STAGE 1: BUILDER (Compile dependencies) 🏗️
# ==========================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for building Python packages
# Removed 'software-properties-common' to avoid apt-get errors on slim images.
# These packages are only needed for building wheels and will be discarded later.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
# Activate the virtual environment by adding it to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: RUNTIME (Production Image) 🎭
# ==========================================
FROM python:3.10-slim as runtime

WORKDIR /app

# SECURITY: Create a non-root user 'appuser'
# Running as root is a security risk; using a dedicated user is best practice.
RUN useradd -m -u 1000 appuser

# Copy the virtual environment from the builder stage
# This excludes build tools (gcc, git, etc.) keeping the image light.
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
# Add 'src' to PYTHONPATH to avoid "Module not found" errors
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Copy application code and grant ownership to 'appuser'
# Critical: If files are owned by root, 'appuser' cannot write to them (e.g., Qdrant storage).
COPY --chown=appuser:appuser . .

# Make the entrypoint script executable
RUN chmod +x run.sh

# Expose the Streamlit port
EXPOSE 8502

# Switch to non-root user
USER appuser

# Start the application (API + Frontend via wrapper script)
CMD ["./run.sh"]