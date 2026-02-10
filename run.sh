#!/bin/bash

# Simplified run script for Intelligent Q&A System
# Sets PYTHONPATH and runs the backend as a module

echo "🚀 Starting Intelligent Q&A System Backend..."

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Set PYTHONPATH to the project root
export PYTHONPATH="$DIR"

# Run the application
# Note: Ensure .env is configured with OPENAI_API_KEY and SECRET_KEY
"$DIR/app/venv/bin/python" -m app.main
