#!/usr/bin/env python3
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite


# ---------------- Config ----------------
VIDEO_DEVICE = "/dev/video0"      # change if needed
MODEL_SIZE = 224
THUMB_WIDTH = 640
WARMUP_FRAMES = 5

BASE_DIR = Path.home() / "wildlife_edge"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = BASE_DIR / "events"

MODEL_PATH = MODELS_DIR / "model_int8.tflite"
LABELS_PATH = MODELS_DIR / "labels.json"


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def center_crop_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_bgr[y0:y0+s, x0:x0+s]


def make_thumbnail(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= width:
        return img_bgr
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def save_jpeg(path: Path, img_bgr: np.ndarray, quality: int = 90):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def prepare_model_input(img_bgr: np.ndarray, size: int, input_detail):
    # Returns:
    #   rgb_uint8: uint8 RGB image for saving/inspection
    #   x_model: tensor in the dtype expected by the model
    sq = center_crop_square(img_bgr)
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    rgb_uint8 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)

    in_dtype = input_detail["dtype"]
    in_scale, in_zero = input_detail["quantization"]

    if in_dtype == np.int8:
        x_model = np.round(x / in_scale + in_zero)
        x_model = np.clip(x_model, -128, 127).astype(np.int8)
    elif in_dtype == np.uint8:
        x_model = np.round(x / in_scale + in_zero)
        x_model = np.clip(x_model, 0, 255).astype(np.uint8)
    else:
        # float model
        x_model = x.astype(np.float32)

    return rgb_uint8, x_model


def load_labels(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    # Ensure dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

    labels = load_labels(LABELS_PATH)

    # Load TFLite interpreter once
    t0_load = time.perf_counter()
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), num_threads=2)
    interpreter.allocate_tensors()
    t1_load = time.perf_counter()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Output quant params for dequant (optional)
    out_scale, out_zero = out["quantization"]  # (scale, zero_point)

    print(f"[ok] Loaded model: {MODEL_PATH.name} in {(t1_load - t0_load)*1000:.1f} ms")
    print(f"[ok] Input dtype={inp['dtype']} shape={inp['shape']}, output dtype={out['dtype']} shape={out['shape']}")
    print("Press ENTER to capture + classify (Ctrl+C to exit).")

    while True:
        input()
        event_id = uuid.uuid4().hex[:12]
        ts = now_iso()
        event_folder = DATA_DIR / event_id
        event_folder.mkdir(parents=True, exist_ok=True)

        full_path = event_folder / "full.jpg"
        thumb_path = event_folder / "thumb.jpg"
        model_in_path = event_folder / f"model_{MODEL_SIZE}.jpg"

        t0 = time.perf_counter()

        # Open camera
        cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {VIDEO_DEVICE}")
        t_open = time.perf_counter()

        # Warmup frames
        for _ in range(WARMUP_FRAMES):
            cap.read()

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame")
        t_cap = time.perf_counter()

        # Save full + thumb
        save_jpeg(full_path, frame, quality=92)
        t_full = time.perf_counter()

        thumb = make_thumbnail(frame, THUMB_WIDTH)
        save_jpeg(thumb_path, thumb, quality=85)
        t_thumb = time.perf_counter()

        # Prepare model input
        rgb_for_save, x = prepare_model_input(frame, MODEL_SIZE, inp)
        # Save model input for inspection
        Image.fromarray(rgb_for_save).save(model_in_path, format="JPEG", quality=95)
        t_prep = time.perf_counter()

        # Inference
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        y_q = interpreter.get_tensor(out["index"])[0]
        t_inf = time.perf_counter()

        # Convert output to float scores for readability
        if np.issubdtype(out["dtype"], np.integer):
            out_scale, out_zero = out["quantization"]
            y = (y_q.astype(np.float32) - out_zero) * out_scale
        else:
            y = y_q.astype(np.float32)

        top3 = np.argsort(y)[-3:][::-1]
        pred = int(top3[0])
        label = labels[pred] if pred < len(labels) else f"class_{pred}"
        conf = float(y[pred])

        record = {
            "event_id": event_id,
            "timestamp": ts,
            "camera": "C270",
            "video_device": VIDEO_DEVICE,
            "paths": {
                "full": str(full_path),
                "thumb": str(thumb_path),
                "model_input": str(model_in_path),
            },
            "classifier": {
                "label": label,
                "confidence": conf,
                "top3": [
                    {"label": labels[int(i)] if int(i) < len(labels) else f"class_{int(i)}",
                     "score": float(y[int(i)])}
                    for i in top3
                ],
                "model": str(MODEL_PATH.name),
            },
            "latency_ms": {
                "open_cam": (t_open - t0) * 1000,
                "capture": (t_cap - t_open) * 1000,
                "save_full": (t_full - t_cap) * 1000,
                "save_thumb": (t_thumb - t_full) * 1000,
                "prep_model_input": (t_prep - t_thumb) * 1000,
                "inference": (t_inf - t_prep) * 1000,
                "end_to_end": (t_inf - t0) * 1000,
            }
        }

        event_json = EVENTS_DIR / f"{event_id}.json"
        event_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

        print(f"\n[event] {event_id}  label={label}  conf={conf:.3f}")
        print("Top-3:")
        for item in record["classifier"]["top3"]:
            print(f"  {item['label']:<16s} {item['score']:.3f}")

        print("Latency (ms):")
        for k, v in record["latency_ms"].items():
            print(f"  {k:<16s} {v:8.2f}")

        print(f"Saved: {full_path}")
        print("Press ENTER for next capture...")

if __name__ == "__main__":
    main()