#!/usr/bin/env python3
import json
import time
import uuid
import requests
import cv2
import numpy as np
import socketio  # Required for the dashboard button
from pathlib import Path
from gpiozero import MotionSensor # Required for the PIR
from ai_edge_litert.interpreter import Interpreter as tflite
import ai_edge_litert.interpreter as _litert

# ---------------- Config ----------------
VIDEO_DEVICE = "/dev/video1"  # Adjusted for Pi 5 USB camera
MODEL_SIZE = 224
THUMB_WIDTH = 640
WARMUP_FRAMES = 10
PIR_PIN = 4 # Connect PIR Data to GPIO 4 (Physical Pin 7)
PI5_IP = "10.0.0.9" # CHANGE TO YOUR PI 5 IP
PI5_URL = f"http://{PI5_IP}:5000"

# Fixed paths
BASE_DIR = Path.home() / "forest_data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "captures"
EVENTS_DIR = BASE_DIR / "events"

MODEL_PATH = MODELS_DIR / "model_int8.tflite"
LABELS_PATH = MODELS_DIR / "labels.json"

# ---------------- Initialization ----------------
sio = socketio.Client()
pir = MotionSensor(PIR_PIN)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
interpreter = tflite(
    model_path=str(MODEL_PATH),
    num_threads=2,
    experimental_op_resolver_type=_litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
out_scale, out_zero = out["quantization"]

# ---------------- Logic ----------------

def capture_and_process():
    """Exactly your original logic moved into a function"""
    event_id = uuid.uuid4().hex[:12]
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n[ACTION] Triggered! Capturing (Event: {event_id})...")
    
    event_folder = DATA_DIR / event_id
    event_folder.mkdir(parents=True, exist_ok=True)

    # Capture Image
    cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Failed to open {VIDEO_DEVICE}")
        return
        
    for _ in range(WARMUP_FRAMES): cap.read()
    ok, frame = cap.read()
    cap.release()
    
    if not ok or frame is None:
        print("Camera read failed, skipping...")
        return

    # Process Images
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = frame[y0:y0+s, x0:x0+s]
    
    resized = cv2.resize(sq, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
    
    # AI Inference
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(rgb.astype(np.uint8), axis=0)
    
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y_q = interpreter.get_tensor(out["index"])[0]
    
    if out["dtype"] == np.uint8:
        y = (y_q.astype(np.float32) - out_zero) * out_scale
    else:
        y = y_q.astype(np.float32)

    top3 = np.argsort(y)[-3:][::-1]
    pred = int(top3[0])
    label = labels[pred] if pred < len(labels) else f"class_{pred}"
    conf = float(y[pred])
    
    print(f"[AI RESULT] {label} (Confidence: {conf:.2f})")

    # Send data to Pi 5 Server
    thumb = cv2.resize(frame, (THUMB_WIDTH, int(h * (THUMB_WIDTH/w))))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    try:
        response = requests.post(f"{PI5_URL}/upload", 
            files={"image": ("img.jpg", buf.tobytes(), "image/jpeg")}, 
            data={
                "timestamp": ts,
                "label": label,
                "confidence": f"{conf:.2f}"
            },
            timeout=5
        )
        print(f"[NETWORK] POST to Pi 5: {response.status_code}")
    except Exception as e:
        print(f"[NETWORK ERROR] Could not reach Pi 5: {e}")

# --- Event Triggers ---

# 1. PIR Sensor Trigger
pir.when_motion = capture_and_process

# 2. Remote Dashboard Button Trigger
@sio.on('remote_capture')
def on_remote_trigger(data):
    capture_and_process()

# --- Main Loop ---
def main():
    try:
        # Connect to Pi 5 for the button listener
        print(f"[info] Connecting to dashboard at {PI5_URL}...")
        sio.connect(PI5_URL)
        
        print("System Armed. Waiting for PIR, Dashboard, or ENTER...")
        while True:
            # 3. Manual Keyboard Trigger
            input("Press ENTER to force a capture...\n")
            capture_and_process()
            
    except KeyboardInterrupt:
        print("\n[info] Exiting...")
        sio.disconnect()

if __name__ == "__main__":
    main()
