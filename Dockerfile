# 1. Base Image
FROM python:3.10-slim

# 2. Create working directory
WORKDIR /app

# 3. Prevent unnecessary files from being created
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (for PostgreSQL and build tools)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all project codes into
COPY . .

# 7. (Optional) The default command, docker-compose, will override this
CMD ["python", "src/app.py"]