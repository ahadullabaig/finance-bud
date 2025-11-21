#!/bin/bash
# Backend runner script - sets up PYTHONPATH and starts the server
cd "$(dirname "$0")"
export PYTHONPATH="."
uvicorn main:app --reload --host 0.0.0.0 --port ${PORT:-8000}
