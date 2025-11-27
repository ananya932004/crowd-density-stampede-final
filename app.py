import io
import os
import base64
from pathlib import Path
import time

from flask import Flask, render_template, request, jsonify
from PIL import Image

from models import detection as detect_mod
from models import density as density_mod

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and detection."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    capacity = request.form.get("capacity", type=float)
    if capacity is None:
        capacity = 50.0

    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    # Read essential options from form
    high_accuracy = request.form.get('high_accuracy', '0') in ('1', 'true', 'True')
    use_density = request.form.get('use_density', '0') in ('1', 'true', 'True')

    # Set defaults for all detection parameters (hardcoded, no UI exposure)
    model_name = 'yolov8n.pt'
    imgsz = 1280
    conf = 0.25
    nms_iou = 0.45
    tile_size = 1024
    overlap = 0.2
    density_threshold = 30.0
    
    # Enable tiling when high_accuracy is requested
    tiling = high_accuracy

    start_time = time.time()
    try:
        counts, annotated_png, kept = detect_mod.detect(
            str(save_path), 
            model_name=model_name, 
            conf=conf, 
            imgsz=imgsz, 
            nms_iou=nms_iou, 
            tiling=tiling, 
            tile_size=tile_size, 
            overlap=overlap
        )
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500

    # Extract people count from detection
    people_count_det = counts.get("person", counts.get("people", 0))
    try:
        people_count_det = float(people_count_det)
    except Exception:
        people_count_det = 0.0

    density_count = None
    density_heatmap_b64 = None
    weights_present = False
    try:
        weights_present = density_mod.weights_present()
    except Exception:
        weights_present = False
    
    density_used = False
    density_reason = None

    if high_accuracy:
        # Run density if user requested or YOLO count >= threshold
        run_density = use_density or (people_count_det >= density_threshold)
        if run_density and weights_present:
            density_used = True
            density_reason = 'user' if use_density else 'threshold'
            try:
                pil = Image.open(save_path).convert('RGB')
                total, heat_bytes, density_map = density_mod.estimate_density(pil, max_side=1024)
                density_count = float(total)
                density_heatmap_b64 = "data:image/png;base64," + base64.b64encode(heat_bytes).decode()
            except Exception as e:
                density_count = None
                density_heatmap_b64 = None
                density_used = False
                density_reason = f'error:{e}'
        else:
            density_used = False
            density_reason = 'weights_missing' if (run_density and not weights_present) else 'not_requested'
    else:
        density_used = False
        density_reason = 'high_accuracy_off'

    # Choose effective people count: prefer density_count when available
    effective_people = density_count if density_count is not None else people_count_det
    try:
        effective_people_f = float(effective_people)
    except Exception:
        effective_people_f = 0.0

    risk = effective_people_f > float(capacity)
    message = ("(stampade chances are hight ,take action)" if risk else "Within safe capacity")

    img_b64 = "data:image/png;base64," + base64.b64encode(annotated_png).decode()
    elapsed_ms = int((time.time() - start_time) * 1000)

    response = {
        "counts": counts,
        "capacity": capacity,
        "people_count_detection": people_count_det,
        "density_count": density_count,
        "effective_people": effective_people,
        "risk": bool(risk),
        "message": message,
        "annotated_image": img_b64,
        "density_heatmap": density_heatmap_b64,
        "debug": {
            "model_name": model_name,
            "imgsz": imgsz,
            "conf": conf,
            "nms_iou": nms_iou,
            "tiling": tiling,
            "tile_size": tile_size,
            "overlap": overlap,
            "inference_ms": elapsed_ms,
            "density_used": density_used,
            "density_reason": density_reason,
            "weights_present": weights_present
        }
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
