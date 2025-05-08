#!/bin/bash
echo "======================================================================="
echo "Demand Forecasting Application - 100% Offline Mode for Unix/Mac"
echo "======================================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in your PATH"
    echo "Please install Python 3 to run this application"
    exit 1
fi

# Run the Python launcher script which handles all the setup
python3 run.py

# Check if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with errors"
    echo "If you're having trouble, try these troubleshooting steps:"
    echo "  1. Make sure streamlit is installed: pip3 install streamlit"
    echo "  2. Make sure all dependencies are installed: pip3 install -r requirements_local.txt"
    echo "  3. Check if another application is using the port"
    echo "  4. Try manually running: streamlit run app.py --server.address 127.0.0.1"
    echo ""
    echo "Press Enter to exit..."
    read
else
    echo "Application closed successfully"
fi