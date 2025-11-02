#!/bin/bash
# Startup script for Web STT Demo
# This script starts both the FastAPI backend and Streamlit frontend

echo "Starting Web Speech-to-Text Demo..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Error: Dependencies not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

echo "Starting FastAPI backend on port 8000..."
python3 ./src/api/api.py &
API_PID=$!

# Wait for API to start
sleep 3

echo "Starting Streamlit frontend on port 8501..."
streamlit run ./src/ui/app.py &
STREAMLIT_PID=$!

echo ""
echo "=========================================="
echo "Web STT Demo is running!"
echo "=========================================="
echo "API: http://localhost:8000"
echo "Web Interface: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Trap Ctrl+C and kill both processes
trap "echo 'Stopping services...'; kill $API_PID $STREAMLIT_PID; exit" INT

# Wait for both processes
wait
