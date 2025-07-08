#!/bin/bash
# Start script for Render deployment

# Install dependencies
pip install -r requirements.txt
 
# Start the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 