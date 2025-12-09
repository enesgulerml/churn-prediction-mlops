# 1. Python 3.10
FROM python:3.10-slim

# 2. Set working folder
WORKDIR /app

# 3. Update system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL files (src, data, qdrant_storage) into
COPY . .

# 6. Show Python the location of the "src" folder
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 7. Open port out
EXPOSE 8502

# 8. Launch the app (Streamlit UI)
CMD ["streamlit", "run", "src/frontend/ui.py", "--server.port=8502", "--server.address=0.0.0.0"]