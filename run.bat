@echo off
echo ======================================================================
echo Demand Forecasting Application - 100%% Offline Mode for Windows
echo ======================================================================
echo.
echo Looking for available port...

REM Use Python to run our app with enhanced error handling
python run.py

echo.
if errorlevel 1 (
    echo Application exited with errors
    echo If you're having trouble, try these troubleshooting steps:
    echo   1. Make sure streamlit is installed: pip install streamlit
    echo   2. Make sure all dependencies are installed: pip install -r requirements_local.txt
    echo   3. Check if another application is using the port
    echo   4. Try manually running: streamlit run app.py --server.address 127.0.0.1
    echo.
    pause
) else (
    echo Application closed successfully
)