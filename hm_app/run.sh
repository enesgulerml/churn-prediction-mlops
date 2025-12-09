#!/bin/bash

# 1. First Start the API (Cook) in Background (&)
echo "🚀 Starting FastAPI Backend..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait 5 seconds for the API to recover (Docker trick)
sleep 5

# 2. Now Start Frontend
echo "🎨 Starting Streamlit Frontend..."
streamlit run src/frontend/ui.py --server.port=8502 --server.address=0.0.0.0