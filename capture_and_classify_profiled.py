#!/usr/bin/env python3
import argparse
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite


# ---------------- Config ----------------
VIDEO_DEVICE = "/dev/video0"
MODEL_SIZE = 224
THUMB_WIDTH = 640
WARMUP_FRAMES = 10

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


def prepare_model_input(img_bgr: np.ndarray, size: int) -> np.ndarray:
    sq = center_crop_square(img_bgr)
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(rgb.astype(np.uint8), axis=0)
    return x


def load_labels(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_one_event(interpreter, inp, out, labels):
    event_id = uuid.uuid4().hex[:12]
    ts = now_iso()
    event_folder = DATA_DIR / event_id
    event_folder.mkdir(parents=True, exist_ok=True)

    full_path = event_folder / "full.jpg"
    thumb_path = event_folder / "thumb.jpg"
    model_in_path = event_folder / f"model_{MODEL_SIZE}.jpg"

    out_scale, out_zero = out["quantization"]

    t0 = time.perf_counter()

    cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {VIDEO_DEVICE}")
    t_open = time.perf_counter()

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to capture frame")
    t_cap = time.perf_counter()

    save_jpeg(full_path, frame, quality=92)
    t_full = time.perf_counter()

    thumb = make_thumbnail(frame, THUMB_WIDTH)
    save_jpeg(thumb_path, thumb, quality=85)
    t_thumb = time.perf_counter()

    x = prepare_model_input(frame, MODEL_SIZE)
    Image.fromarray(x[0]).save(model_in_path, format="JPEG", quality=95)
    t_prep = time.perf_counter()

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y_q = interpreter.get_tensor(out["index"])[0]
    t_inf = time.perf_counter()

    if out["dtype"] == np.uint8:
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
                {
                    "label": labels[int(i)] if int(i) < len(labels) else f"class_{int(i)}",
                    "score": float(y[int(i)])
                }
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

    return record


def run_manual_mode(interpreter, inp, out, labels):
    print("Press ENTER to capture + classify (Ctrl+C to exit).")
    while True:
        input()
        run_one_event(interpreter, inp, out, labels)
        print("Press ENTER for next capture...")


def run_batch_mode(interpreter, inp, out, labels, num_events: int, interval_s: float):
    print(f"[mode] Batch mode: {num_events} event(s), {interval_s:.1f}s apart")
    for i in range(num_events):
        print(f"\n--- Event {i + 1}/{num_events} ---")
        event_start = time.perf_counter()
        run_one_event(interpreter, inp, out, labels)
        elapsed = time.perf_counter() - event_start
        sleep_time = max(0.0, interval_s - elapsed)

        if i < num_events - 1 and sleep_time > 0:
            print(f"[wait] Sleeping {sleep_time:.2f}s before next event")
            time.sleep(sleep_time)

    print("\n[done] Batch mode complete.")


def run_timed_mode(interpreter, inp, out, labels, duration_s: float, interval_s: float):
    print(f"[mode] Timed mode: {duration_s:.1f}s total, {interval_s:.1f}s apart")
    overall_start = time.perf_counter()
    event_count = 0

    while True:
        if time.perf_counter() - overall_start >= duration_s:
            break

        event_count += 1
        print(f"\n--- Event {event_count} ---")
        event_start = time.perf_counter()
        run_one_event(interpreter, inp, out, labels)
        elapsed = time.perf_counter() - event_start
        sleep_time = max(0.0, interval_s - elapsed)

        if time.perf_counter() - overall_start >= duration_s:
            break

        if sleep_time > 0:
            print(f"[wait] Sleeping {sleep_time:.2f}s before next event")
            time.sleep(sleep_time)

    total_time = time.perf_counter() - overall_start
    print(f"\n[done] Timed mode complete. Ran {event_count} event(s) in {total_time:.2f}s.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="Run exactly N events")
    parser.add_argument("--duration", type=float, default=0, help="Run for N seconds")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between event starts")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

    if args.batch > 0 and args.duration > 0:
        raise ValueError("Use either --batch or --duration, not both.")

    labels = load_labels(LABELS_PATH)

    t0_load = time.perf_counter()
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), num_threads=2)
    interpreter.allocate_tensors()
    t1_load = time.perf_counter()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"[ok] Loaded model: {MODEL_PATH.name} in {(t1_load - t0_load)*1000:.1f} ms")
    print(f"[ok] Input dtype={inp['dtype']} shape={inp['shape']}, output dtype={out['dtype']} shape={out['shape']}")

    if args.batch > 0:
        run_batch_mode(interpreter, inp, out, labels, args.batch, args.interval)
    elif args.duration > 0:
        run_timed_mode(interpreter, inp, out, labels, args.duration, args.interval)
    else:
        run_manual_mode(interpreter, inp, out, labels)


if __name__ == "__main__":
    main()
    