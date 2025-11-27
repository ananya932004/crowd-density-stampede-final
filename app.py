import io
import os
import base64
from pathlib import Path
import time

from flask import Flask, render_template, request, jsonify

from PIL import Image

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# Import detection and density modules (they handle missing deps themselves)
from models import detection as detect_mod
from models import density as density_mod


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Expect 'image' file and optional 'capacity' form field
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # capacity
    capacity = request.form.get("capacity", type=float)
    if capacity is None:
        capacity = 50.0  # default

    # Save uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    # Read advanced options
    high_accuracy = request.form.get('high_accuracy', '0') in ('1', 'true', 'True')
    model_name = request.form.get('model_name', 'yolov8n.pt')
    imgsz = request.form.get('imgsz', type=int)
    conf = request.form.get('conf', type=float) or 0.25
    nms_iou = request.form.get('nms_iou', type=float) or 0.45
    tiling = request.form.get('tiling', '0') in ('1', 'true', 'True')
    tile_size = request.form.get('tile_size', type=int) or 1024
    overlap = request.form.get('overlap', type=float) or 0.2

    # If high_accuracy requested but tiling not explicitly set, enable tiling
    if high_accuracy and not tiling:
        tiling = True

    start_time = time.time()
    try:
        counts, annotated_png, kept = detect_mod.detect(str(save_path), model_name=model_name, conf=conf, imgsz=imgsz, nms_iou=nms_iou, tiling=tiling, tile_size=tile_size, overlap=overlap)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500

    people_count_det = counts.get("person", counts.get("people", 0))
    try:
        people_count_det = float(people_count_det)
    except Exception:
        people_count_det = 0.0

    density_count = None
    density_heatmap_b64 = None
    # Run density estimator when high_accuracy is enabled
    weights_present = False
    try:
        weights_present = density_mod.weights_present()
    except Exception:
        weights_present = False

    density_used = False
    density_reason = None

    if high_accuracy:
        # decide whether to run density based on user toggle or threshold
        use_density_flag = request.form.get('use_density', '0') in ('1', 'true', 'True')
        density_threshold = request.form.get('density_threshold', type=float) or 30.0
        # run density if user requested or YOLO count >= threshold
        run_density = use_density_flag or (people_count_det >= density_threshold)
        if run_density and weights_present:
            density_used = True
            density_reason = 'user' if use_density_flag else 'threshold'
            try:
                pil = Image.open(save_path).convert('RGB')
                # use smaller side for faster CPU inference
                total, heat_bytes, density_map = density_mod.estimate_density(pil, max_side=1024)
                density_count = float(total)
                density_heatmap_b64 = "data:image/png;base64," + base64.b64encode(heat_bytes).decode()
            except Exception as e:
                # don't fail the whole request for missing torch/weights
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
    message = ("stampade chances are high, take action)" if risk else "Within safe capacity, no chances of stampede")

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
        "debug": {"model_name": model_name, "imgsz": imgsz, "conf": conf, "nms_iou": nms_iou, "tiling": tiling, "tile_size": tile_size, "overlap": overlap, "inference_ms": elapsed_ms, "density_used": density_used, "density_reason": density_reason, "weights_present": weights_present}
    }

    return jsonify(response)


if __name__ == "__main__":
    # Run in debug mode on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
import io
import os
import base64
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Try to import Ultralytics YOLO. The first run will download weights (yolov8n.pt).
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

from PIL import Image
import numpy as np

from models import detection as detect_mod
from models import density as density_mod
import time

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# Load models lazily to avoid long startup if not needed
MODELS = {}


def load_model(name="yolov8n.pt"):
    global MODELS
    if name in MODELS:
        return MODELS[name]
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics YOLO is not available. Install requirements from requirements.txt"
        )
    model = YOLO(name)
    MODELS[name] = model
    return model


def iou(box_a, box_b):
    # boxes are [x1,y1,x2,y2]
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def dedupe_boxes(boxes, scores, classes, iou_thresh=0.45):
    # Greedy NMS-style deduplication: keep highest score, remove boxes with IoU>threshold
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in idxs:
        b = boxes[i]
        skip = False
        for j in keep:
            if classes[i] != classes[j]:
                continue
            if iou(b, boxes[j]) > iou_thresh:
                skip = True
                break
        if not skip:
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep]


def detect_and_annotate(image_path, model_name="yolov8n.pt", conf=0.25, imgsz=None, nms_iou=0.45, tiling=False, tile_size=1024, overlap=0.2):
    """Run detection on the image and return (counts_dict, annotated_png_bytes)

    Supports optional tiling mode for higher-accuracy at the cost of compute.
    """
    model = load_model(model_name)
    # Load image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    all_boxes = []
    all_scores = []
    all_classes = []

    if tiling:
        # compute tile grid
        step = int(tile_size * (1 - overlap))
        xs = list(range(0, max(1, W - tile_size + 1), step))
        ys = list(range(0, max(1, H - tile_size + 1), step))
        # ensure last tile covers the edge
        if xs == []:
            xs = [0]
        if ys == []:
            ys = [0]
        if xs[-1] + tile_size < W:
            xs.append(max(0, W - tile_size))
        if ys[-1] + tile_size < H:
            ys.append(max(0, H - tile_size))

        for y in ys:
            for x in xs:
                box = (x, y, min(x + tile_size, W), min(y + tile_size, H))
                tile = img.crop(box)
                # Convert to numpy array BGR if needed, but ultralytics accepts PIL
                try:
                    # pass imgsz if provided
                    if imgsz:
                        results = model(tile, conf=conf, imgsz=imgsz)
                    else:
                        results = model(tile, conf=conf)
                except Exception:
                    results = model(tile, conf=conf)
                res = results[0]
                # extract boxes, scores, classes
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback for older versions
                    boxes = []
                    scores = []
                    classes = []
                    for b in res.boxes:
                        try:
                            boxes.append([b.x1, b.y1, b.x2, b.y2])
                            scores.append(float(b.conf))
                            classes.append(int(b.cls))
                        except Exception:
                            pass
                    boxes = np.array(boxes)
                    scores = np.array(scores)
                    classes = np.array(classes)

                # shift box coords by x,y
                for i, b in enumerate(boxes):
                    shifted = [b[0] + x, b[1] + y, b[2] + x, b[3] + y]
                    all_boxes.append(shifted)
                    all_scores.append(float(scores[i]))
                    all_classes.append(int(classes[i]))

        # dedupe boxes
        if len(all_boxes) > 0:
            kept_boxes, kept_scores, kept_classes = dedupe_boxes(all_boxes, all_scores, all_classes, iou_thresh=nms_iou)
        else:
            kept_boxes, kept_scores, kept_classes = [], [], []

        # Prepare annotated image by drawing boxes on original image
        draw = Image.fromarray(np.array(img))
        draw_ctx = Image.Image.draw if False else None
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)

        names = res.names if hasattr(res, "names") else model.names
        counts = {}
        for i, b in enumerate(kept_boxes):
            cls = kept_classes[i]
            name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            counts[name] = counts.get(name, 0) + 1
            # draw rectangle
            draw.rectangle(((b[0], b[1]), (b[2], b[3])), outline="red", width=2)
            draw.text((b[0], b[1] - 10), f"{name} {kept_scores[i]:.2f}", fill="red")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return counts, png_bytes

    # non-tiling path (original behavior)
    try:
        if imgsz:
            results = model(image_path, conf=conf, imgsz=imgsz)
        else:
            results = model(image_path, conf=conf)
    except Exception as e:
        results = model(image_path, conf=conf)
    res = results[0]

    # res.boxes.cls is an array of class indices; res.names maps ids to names
    names = res.names if hasattr(res, "names") else model.names

    counts = {}
    try:
        # For newer ultralytics versions, res.boxes.cls exists
        classes = [int(c) for c in res.boxes.cls.cpu().numpy().flatten()]
    except Exception:
        # Fallback: iterate res.boxes.data if available
        classes = []
        try:
            for b in res.boxes:
                classes.append(int(b.cls))
        except Exception:
            classes = []

    for c in classes:
        name = names.get(c, str(c)) if isinstance(names, dict) else names[c]
        counts[name] = counts.get(name, 0) + 1

    # Generate annotated image (Ultralytics provides a plot() utility on result)
    try:
        annotated = res.plot()
        # res.plot() returns an ndarray (BGR) or PIL image depending on version
        if isinstance(annotated, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            if annotated.shape[2] == 3:
                annotated = annotated[..., ::-1]
            img_out = Image.fromarray(annotated)
        else:
            img_out = Image.fromarray(np.array(annotated))
    except Exception:
        # Fallback: just return original image
        img_out = Image.open(image_path).convert("RGB")

    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return counts, png_bytes
    """Run detection on the image and return (counts_dict, annotated_png_bytes)

    counts_dict maps class name to integer count.
    """
    model = load_model()
    # Run prediction
    results = model(image_path, conf=conf)
    # Use first result
    res = results[0]

    # res.boxes.cls is an array of class indices; res.names maps ids to names
    names = res.names if hasattr(res, "names") else model.names

    counts = {}
    try:
        # For newer ultralytics versions, res.boxes.cls exists
        classes = [int(c) for c in res.boxes.cls.cpu().numpy().flatten()]
    except Exception:
        # Fallback: iterate res.boxes.data if available
        classes = []
        try:
            for b in res.boxes:
                classes.append(int(b.cls))
        except Exception:
            classes = []

    for c in classes:
        name = names.get(c, str(c)) if isinstance(names, dict) else names[c]
        counts[name] = counts.get(name, 0) + 1

    # Generate annotated image (Ultralytics provides a plot() utility on result)
    try:
        annotated = res.plot()
        # res.plot() returns an ndarray (BGR) or PIL image depending on version
        if isinstance(annotated, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            if annotated.shape[2] == 3:
                annotated = annotated[..., ::-1]
            img = Image.fromarray(annotated)
        else:
            img = Image.fromarray(np.array(annotated))
    except Exception:
        # Fallback: just return original image
        img = Image.open(image_path).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return counts, png_bytes


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Expect 'image' file and optional 'capacity' form field
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    capacity = request.form.get("capacity", type=int)
    if capacity is None:
        capacity = 50  # default

    # Save uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    # Read advanced options
    high_accuracy = request.form.get('high_accuracy', '0') in ('1', 'true', 'True')
    model_name = request.form.get('model_name', 'yolov8n.pt')
    imgsz = request.form.get('imgsz', type=int)
    conf = request.form.get('conf', type=float) or 0.25
    nms_iou = request.form.get('nms_iou', type=float) or 0.45
    tiling = request.form.get('tiling', '0') in ('1', 'true', 'True')
    tile_size = request.form.get('tile_size', type=int) or 1024
    overlap = request.form.get('overlap', type=float) or 0.2

    start_time = time.time()
    try:
        counts, annotated_png, kept = detect_mod.detect(str(save_path), model_name=model_name, conf=conf, imgsz=imgsz, nms_iou=nms_iou, tiling=tiling, tile_size=tile_size, overlap=overlap)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500

    people_count = counts.get("person", counts.get("people", 0))
    try:
        people_count = int(people_count)
    except Exception:
        people_count = 0

    density_count = None
    density_heatmap_b64 = None
    # Run density estimator when high_accuracy is enabled
    if high_accuracy:
        try:
            pil = Image.open(save_path).convert('RGB')
            total, heat_bytes, density_map = density_mod.estimate_density(pil)
            density_count = float(total)
            density_heatmap_b64 = "data:image/png;base64," + base64.b64encode(heat_bytes).decode()
        except Exception as e:
            # don't fail the whole request for missing torch/weights
            density_count = None
            density_heatmap_b64 = None

    risk = False
    # prefer density_count when available for people count
    effective_people = density_count if density_count is not None else people_count
    try:
        effective_people_f = float(effective_people)
    except Exception:
        effective_people_f = 0.0
    risk = effective_people_f > capacity

    message = ("stampade chances are high, take action" if risk else "Within safe capacity, no chances of stampede")

    img_b64 = "data:image/png;base64," + base64.b64encode(annotated_png).decode()
    elapsed_ms = int((time.time() - start_time) * 1000)

    response = {
        "counts": counts,
        "capacity": capacity,
        "people_count": people_count,
        "density_count": density_count,
        "effective_people": effective_people,
        "risk": bool(risk),
        "message": message,
        "annotated_image": img_b64,
        "density_heatmap": density_heatmap_b64,
        "debug": {"model_name": model_name, "imgsz": imgsz, "conf": conf, "nms_iou": nms_iou, "tiling": tiling, "tile_size": tile_size, "overlap": overlap, "inference_ms": elapsed_ms}
    }

    return jsonify(response)


if __name__ == "__main__":
    # Run in debug mode on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
