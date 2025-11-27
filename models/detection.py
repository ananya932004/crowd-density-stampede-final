"""
Detection wrapper for YOLO model loading and tiled / non-tiled inference.
This file refactors detection logic out of app.py so `app.py` can orchestrate both detectors.
"""
from pathlib import Path
import io
from PIL import Image, ImageDraw
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

MODELS = {}


def load_yolo(name="yolov8n.pt"):
    global MODELS
    if name in MODELS:
        return MODELS[name]
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is required for detection. Install ultralytics.")
    model = YOLO(name)
    MODELS[name] = model
    return model


def iou(box_a, box_b):
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


def detect(image_path_or_pil, model_name="yolov8n.pt", conf=0.25, imgsz=None, nms_iou=0.45, tiling=False, tile_size=1024, overlap=0.2):
    """Detect objects. Accepts either a path or a PIL image.
    Returns counts dict and annotated PNG bytes, along with raw kept boxes list.
    """
    model = load_yolo(model_name)

    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert('RGB')
    else:
        img = image_path_or_pil.convert('RGB')
    W, H = img.size

    all_boxes = []
    all_scores = []
    all_classes = []

    if tiling:
        step = int(tile_size * (1 - overlap))
        xs = list(range(0, max(1, W - tile_size + 1), step))
        ys = list(range(0, max(1, H - tile_size + 1), step))
        if xs == []:
            xs = [0]
        if ys == []:
            ys = [0]
        if xs[-1] + tile_size < W:
            xs.append(max(0, W - tile_size))
        if ys[-1] + tile_size < H:
            ys.append(max(0, H - tile_size))

        last_res = None
        for y in ys:
            for x in xs:
                tile = img.crop((x, y, min(x + tile_size, W), min(y + tile_size, H)))
                try:
                    if imgsz:
                        results = model(tile, conf=conf, imgsz=imgsz)
                    else:
                        results = model(tile, conf=conf)
                except Exception:
                    results = model(tile, conf=conf)
                res = results[0]
                last_res = res
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                except Exception:
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

                for i, b in enumerate(boxes):
                    shifted = [b[0] + x, b[1] + y, b[2] + x, b[3] + y]
                    all_boxes.append(shifted)
                    all_scores.append(float(scores[i]))
                    all_classes.append(int(classes[i]))

        if len(all_boxes) > 0:
            kept_boxes, kept_scores, kept_classes = dedupe_boxes(all_boxes, all_scores, all_classes, iou_thresh=nms_iou)
        else:
            kept_boxes, kept_scores, kept_classes = [], [], []

        # draw
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        names = last_res.names if last_res is not None and hasattr(last_res, 'names') else model.names
        counts = {}
        kept = []
        for i, b in enumerate(kept_boxes):
            cls = kept_classes[i]
            name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            counts[name] = counts.get(name, 0) + 1
            draw.rectangle(((b[0], b[1]), (b[2], b[3])), outline="red", width=2)
            draw.text((b[0], b[1] - 10), f"{name} {kept_scores[i]:.2f}", fill="red")
            kept.append({'box': b, 'score': kept_scores[i], 'class': cls})

        buf = io.BytesIO()
        draw_img.save(buf, format='PNG')
        return counts, buf.getvalue(), kept

    # non-tiling
    try:
        if imgsz:
            results = model(image_path_or_pil, conf=conf, imgsz=imgsz)
        else:
            results = model(image_path_or_pil, conf=conf)
    except Exception:
        results = model(image_path_or_pil, conf=conf)
    res = results[0]
    names = res.names if hasattr(res, 'names') else model.names

    counts = {}
    kept = []
    try:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for i, b in enumerate(boxes):
            cls = int(classes[i])
            name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            counts[name] = counts.get(name, 0) + 1
            kept.append({'box': b.tolist(), 'score': float(scores[i]), 'class': cls})
    except Exception:
        for b in res.boxes:
            try:
                cls = int(b.cls)
                name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
                counts[name] = counts.get(name, 0) + 1
                kept.append({'box': [b.x1, b.y1, b.x2, b.y2], 'score': float(b.conf), 'class': cls})
            except Exception:
                pass

    try:
        annotated = res.plot()
        if isinstance(annotated, np.ndarray):
            if annotated.shape[2] == 3:
                annotated = annotated[..., ::-1]
            img_out = Image.fromarray(annotated)
        else:
            img_out = Image.fromarray(np.array(annotated))
    except Exception:
        img_out = img

    buf = io.BytesIO()
    img_out.save(buf, format='PNG')
    return counts, buf.getvalue(), kept
