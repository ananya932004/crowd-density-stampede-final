# Crowd Density and Stampede Prediction

Simple Flask web app to upload an image, detect objects (people, cars, cats, etc.) using Ultralytics YOLO, display counts, and warn when the number of people exceeds a configurable capacity threshold (possible stampede risk).

## Quick start

1. Create and activate a Python environment (recommended):

   - Windows (PowerShell):

     ```powershell
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```

2. Run the app:

   ```python3 --version 
   python3 -m venv .venv 
   source .venv/bin/activate 
   pip install -r requirements.txt 
   flask run --port=5002

   powershell
   python app.py
   ```

3. Open http://127.0.0.1:5000 in your browser.

On the first run the YOLO model weights (`yolov8n.pt`) will be downloaded automatically by Ultralytics.

## How it works

- Upload an image in the web UI.
- The backend runs YOLO detection and counts detected classes.
- If the number of people exceeds the configured capacity, an alert message is shown.

## High Accuracy mode

This app includes a "High accuracy mode" toggle in the UI. When enabled it will:
- Optionally use a larger YOLO model (e.g., `yolov8x`) for better detection accuracy.
- Increase inference resolution (configurable, default 1280 on the long side).
- Use tiled inference (split image into overlapping tiles) to detect small/remote people in dense crowds and then deduplicate overlapping detections.

Trade-offs: High accuracy mode is significantly slower and uses more memory. Expect downloads of larger model weights (yolov8l/x) and longer inference times; use a machine with more RAM or a GPU for best performance.

## Notes & next steps

- This is a minimal demo. For production, consider using a GPU-enabled server and more robust model management.
- Add authentication, persistent logging, streaming video detection (for real-time monitoring), and persistent thresholds per area.
