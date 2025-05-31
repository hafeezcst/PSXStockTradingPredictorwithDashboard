import subprocess
import sys
import os
from threading import Thread
import time

def run_backend():
    """Run the FastAPI backend server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"])

def run_frontend():
    """Run the Streamlit frontend"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"])

if __name__ == "__main__":
    # Start backend in a separate thread
    backend_thread = Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()

    # Wait for backend to start
    time.sleep(2)

    # Start frontend
    run_frontend() 