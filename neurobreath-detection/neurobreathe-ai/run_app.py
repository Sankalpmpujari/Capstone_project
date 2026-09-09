"""
NeuroBreathe AI - Unified Application Launcher
Starts the FastAPI server with Uvicorn and automatically launches the web browser.
"""

import os
import sys
import time
import socket
import webbrowser
import threading
from pathlib import Path

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def find_available_port(starting_port=8000, max_attempts=10):
    for p in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return starting_port

def open_browser_delayed(url, delay=1.5):
    time.sleep(delay)
    print(f"\n[NeuroBreathe AI] Opening web browser at: {url}")
    webbrowser.open(url)

def main():
    print("=" * 65)
    print("  NEUROBREATHE AI — CLINICAL DIAGNOSTIC PLATFORM")
    print("  Parkinson's Voice Biomarkers + Respiratory Acoustic AI")
    print("=" * 65)

    port = find_available_port(8000)
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print(f"\n[Server] Binding to http://{host}:{port}")
    print(f"[Status] Initializing models and endpoints...")

    from backend.models_loader import models
    models.initialize()
    status = models.get_status()
    park_ok = "READY" if status["parkinson"]["ready"] else "DEGRADED"
    resp_ok = "READY" if status["respiratory"]["ready"] else "DEGRADED"
    print(f"[Models] Parkinson Model: {park_ok} | Respiratory Model: {resp_ok}")

    # Open browser in a background thread
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
