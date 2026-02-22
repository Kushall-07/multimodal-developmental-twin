#!/usr/bin/env bash
set -e

# Start FastAPI in background
uvicorn app.api.main:app --host 127.0.0.1 --port 8001 &

# Start Streamlit
streamlit run frontend/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 7860 \
  --server.headless true