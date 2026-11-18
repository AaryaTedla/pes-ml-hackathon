@echo off
REM Run from anywhere. Starts the PathSnap API on all network interfaces, port 8000.
REM Optional: set PATHSNAP_TRIPS_DIR=C:\path\to\evaluator_trips before running.
cd /d "%~dp0.."
echo API will be reachable at http://^<this-laptop-IP^>:8000   (find the IP with: ipconfig)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
