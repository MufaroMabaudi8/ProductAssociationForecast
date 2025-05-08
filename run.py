#!/usr/bin/env python3
"""
Run script for Demand Forecasting Application
This script launches the Streamlit app with proper server configuration for offline use
"""
import os
import sys
import subprocess
import socket
import webbrowser
import time
import platform

def check_port_available(port):
    """Check if a port is available to use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    available = True
    try:
        sock.bind(('127.0.0.1', port))
    except:
        available = False
    finally:
        sock.close()
    return available

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port"""
    for attempt in range(max_attempts):
        port = start_port + attempt
        if check_port_available(port):
            return port
    return None

def main():
    """Run the Streamlit application with optimized settings for offline use"""
    print("="*70)
    print("Demand Forecasting Application - 100% Offline Mode")
    print("="*70)
    
    # Find an available port
    port = find_available_port(8501)
    if not port:
        print("ERROR: Could not find an available port. Please close other applications and try again.")
        return 1
    
    # Use 127.0.0.1 explicitly (localhost) to ensure local access works on all systems
    address = "127.0.0.1"
    
    # Run Streamlit with offline-optimized settings
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", address,
        "--server.enableCORS", "false", 
        "--server.enableXsrfProtection", "false",
        "--server.enableWebsocketCompression", "false",
        "--browser.serverAddress", "localhost",
        "--server.headless", "true",
        "--global.developmentMode", "false"
    ]
    
    url = f"http://{address}:{port}"
    
    print(f"\nStarting application in OFFLINE mode...")
    print(f"\nThe application will be available at: {url}")
    print(f"Your system: {platform.system()} {platform.release()}")
    print("\nPlease wait while the application starts...")
    
    # Start the process
    process = None  # Initialize process variable
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start (up to 15 seconds)
        server_started = False
        for _ in range(15):
            time.sleep(1)
            try:
                # Try to connect to the server
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(1)
                test_socket.connect((address, port))
                test_socket.close()
                server_started = True
                break
            except:
                print(".", end="", flush=True)
        
        print()  # New line after dots
        
        if server_started:
            print("\n✓ Application started successfully!")
            print(f"\nAccess the application at: {url}")
            print("Using one of these methods:")
            print(f"  1. Click here: {url}")
            print(f"  2. Copy and paste this URL in your browser: {url}")
            print(f"  3. Type 'localhost:{port}' in your browser")
            
            # Try to open in browser
            try:
                webbrowser.open(url)
                print("\n✓ Attempting to open in your default browser")
            except:
                print("\n✗ Could not open browser automatically")
                
            print("\nPress Ctrl+C to stop the application when finished")
            
            # Keep the script running while the server is running
            process.wait()
        else:
            print("\n✗ Application failed to start properly")
            print("Please check that no other applications are using port", port)
            if process:
                process.terminate()
            return 1
            
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        if process:
            process.terminate()
    except Exception as e:
        print(f"\nError starting application: {e}")
        if process:
            process.terminate()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())