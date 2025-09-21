#!/bin/bash

# Set the port from environment variable (Render provides this)
export PORT=${PORT:-8501}

# Run Streamlit with appropriate configuration for Render
streamlit run stock_predictor.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false