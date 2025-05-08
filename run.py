#!/usr/bin/env python3
"""
Run script for Demand Forecasting Application
This script launches the Streamlit app with proper server configuration
"""
import os
import sys
import subprocess

def main():
    """Run the Streamlit application with optimized settings"""
    print("Starting Demand Forecasting Application...")
    
    # Run Streamlit with optimized settings for external access
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", "8501",  # Standard Streamlit port
        "--server.address", "localhost",  # Use localhost for local access
        "--server.enableCORS", "false", 
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"Execute the following command to run the application locally:")
    print(" ".join(cmd))
    print("\nThe application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        # Run the Streamlit command
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"\nError starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())