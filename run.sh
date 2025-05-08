#!/bin/bash
echo "Starting Demand Forecasting Application..."
echo ""
echo "The application will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py --server.port 8501 --server.address localhost --server.enableCORS false --server.enableXsrfProtection false