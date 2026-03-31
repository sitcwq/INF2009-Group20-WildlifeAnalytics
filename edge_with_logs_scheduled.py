#!/usr/bin/env python3
import argparse
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psutil

import cv2
import numpy as np
import requests
import socketio
from gpiozero import MotionSensor
from PIL import Image
from ai_edge_litert.interpreter import Interpreter as tflite
import ai_edge_litert.interpreter as _litert

# ---------------- Config ----------------
VIDEO_DEVICE = "/dev/video0"  # Adjust as needed
MODEL_SIZE = 224
THUMB_WIDTH = 640
WARMUP_FRAMES = 2
PIR_PIN = 4                     # PIR Data -> GPIO 4 (Pin 7)
PI5_IP = "192.168.1.7"         # CHANGE TO YOUR PI 5 IP
PI5_URL = f"http://{PI5_IP}:5000"
CONNECT_DASHBOARD = True

DEFAULT_PIR_COOLDOWN_S = 2.0
DEFAULT_PIR_FOLLOWUP_DELAY_S = 0.5
DEFAULT_PIR_MAX_FOLLOWUPS = 1

# Paths
BASE_DIR = Path.home() / "wildlife_edge"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = BASE_DIR / "events"
RUN_LOGS_DIR = BASE_DIR / "run_logs"

MODEL_PATH = MODELS_DIR / "model_int8.tflite"
LABELS_PATH = MODELS_DIR / "labels.json"

# ---------------- Utilities ----------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def center_crop_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_bgr[y0:y0 + s, x0:x0 + s]


def make_thumbnail(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= width:
        return img_bgr
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def save_jpeg(path: Path, img_bgr: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def load_labels(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_model_input(img_bgr: np.ndarray, size: int, input_detail):
    """
    Returns:
      rgb_uint8: uint8 RGB image for saving/inspection
      x_model: tensor in the dtype/quantization expected by the model
    """
    sq = center_crop_square(img_bgr)
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    rgb_uint8 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)

    in_dtype = input_detail["dtype"]
    in_scale, in_zero = input_detail["quantization"]

    if in_dtype == np.int8:
        if in_scale == 0:
            raise ValueError("Input quantization scale is 0 for int8 model")
        x_model = np.round(x / in_scale + in_zero)
        x_model = np.clip(x_model, -128, 127).astype(np.int8)
    elif in_dtype == np.uint8:
        if in_scale and in_scale != 0:
            x_model = np.round(x / in_scale + in_zero)
            x_model = np.clip(x_model, 0, 255).astype(np.uint8)
        else:
            x_model = x.astype(np.uint8)
    else:
        x_model = x.astype(np.float32)

    return rgb_uint8, x_model


def dequantize_output(y_raw: np.ndarray, output_detail) -> np.ndarray:
    if np.issubdtype(output_detail["dtype"], np.integer):
        out_scale, out_zero = output_detail["quantization"]
        return (y_raw.astype(np.float32) - out_zero) * out_scale
    return y_raw.astype(np.float32)


def sample_system_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "timestamp": now_iso(),
        "unix_time": time.time(),
        "cpu_percent_total": float(psutil.cpu_percent(interval=None)),
        "cpu_percent_per_core": [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)],
        "memory": {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "free_bytes": int(vm.free),
            "percent": float(vm.percent),
        },
        "process": {
            "pid": int(PROCESS.pid),
            "rss_bytes": int(PROCESS.memory_info().rss),
            "vms_bytes": int(PROCESS.memory_info().vms),
            "cpu_percent": float(PROCESS.cpu_percent(interval=None)),
        },
    }


def compute_system_delta(before: dict, after: dict) -> dict:
    return {
        "memory_used_bytes": after["memory"]["used_bytes"] - before["memory"]["used_bytes"],
        "memory_percent": after["memory"]["percent"] - before["memory"]["percent"],
        "process_rss_bytes": after["process"]["rss_bytes"] - before["process"]["rss_bytes"],
        "process_vms_bytes": after["process"]["vms_bytes"] - before["process"]["vms_bytes"],
        "process_cpu_percent": after["process"]["cpu_percent"] - before["process"]["cpu_percent"],
    }


class SystemSampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = max(0.1, float(interval_s))
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _collect_sample(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "timestamp": now_iso(),
            "unix_time": time.time(),
            "cpu_percent_total": float(psutil.cpu_percent(interval=None)),
            "cpu_percent_per_core": [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)],
            "memory": {
                "available_bytes": int(vm.available),
                "used_bytes": int(vm.used),
                "percent": float(vm.percent),
            },
            "process": {
                "pid": int(PROCESS.pid),
                "rss_bytes": int(PROCESS.memory_info().rss),
                "vms_bytes": int(PROCESS.memory_info().vms),
                "cpu_percent": float(PROCESS.cpu_percent(interval=None)),
            },
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._collect_sample())
            if self._stop.wait(self.interval_s):
                break

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        PROCESS.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def write_run_log(run_id: str, mode: str, started_at: str, completed_at: str, args: dict, events: list, sampler: SystemSampler) -> Path:
    record = {
        "run_id": run_id,
        "mode": mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "args": args,
        "event_count": len(events),
        "events": events,
        "sampler": {
            "interval_s": sampler.interval_s,
            "sample_count": len(sampler.samples),
            "samples": sampler.samples,
        },
    }
    run_log_path = RUN_LOGS_DIR / f"{run_id}_{mode}.json"
    run_log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run_log_path


# ---------------- Runtime Setup ----------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

labels = load_labels(LABELS_PATH)
PROCESS = psutil.Process()
psutil.cpu_percent(interval=None)
psutil.cpu_percent(interval=None, percpu=True)
PROCESS.cpu_percent(interval=None)

interpreter = tflite(
    model_path=str(MODEL_PATH),
    num_threads=2,
    experimental_op_resolver_type=_litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

HTTP = requests.Session()
sio = socketio.Client()
pir = MotionSensor(PIR_PIN)
PIPELINE_LOCK = threading.Lock()
PIR_CONTROLLER = None


# ---------------- Camera / Inference ----------------
def capture_frame() -> np.ndarray:
    cap = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {VIDEO_DEVICE}")

    try:
        for _ in range(WARMUP_FRAMES):
            cap.read()
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to capture frame")

    return frame


def classify_frame(frame: np.ndarray):
    rgb_for_save, x_model = prepare_model_input(frame, MODEL_SIZE, inp)

    interpreter.set_tensor(inp["index"], x_model)
    interpreter.invoke()
    y_raw = interpreter.get_tensor(out["index"])[0]
    y = dequantize_output(y_raw, out)

    top3 = np.argsort(y)[-3:][::-1]
    pred = int(top3[0])
    label = labels[pred] if pred < len(labels) else f"class_{pred}"
    conf = float(y[pred])

    return {
        "label": label,
        "confidence": conf,
        "top3": [
            {
                "label": labels[int(i)] if int(i) < len(labels) else f"class_{int(i)}",
                "score": float(y[int(i)]),
            }
            for i in top3
        ],
        "rgb_model_input": rgb_for_save,
    }


# ---------------- Persistence / Networking ----------------
def save_event_artifacts(event_folder: Path, frame: np.ndarray, rgb_model_input: np.ndarray):
    full_path = event_folder / "full.jpg"
    thumb_path = event_folder / "thumb.jpg"
    model_in_path = event_folder / f"model_{MODEL_SIZE}.jpg"

    save_jpeg(full_path, frame, quality=92)
    thumb = make_thumbnail(frame, THUMB_WIDTH)
    save_jpeg(thumb_path, thumb, quality=85)
    Image.fromarray(rgb_model_input).save(model_in_path, format="JPEG", quality=95)

    return {
        "full": full_path,
        "thumb": thumb_path,
        "model_input": model_in_path,
        "thumb_img": thumb,
    }


def upload_to_pi5(ts: str, label: str, conf: float, thumb_bgr: np.ndarray) -> Optional[int]:
    ok, buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode thumbnail for upload")

    response = HTTP.post(
        f"{PI5_URL}/upload",
        files={"image": ("img.jpg", buf.tobytes(), "image/jpeg")},
        data={
            "timestamp": ts,
            "label": label,
            "confidence": f"{conf:.2f}",
        },
        timeout=5,
    )
    return response.status_code


def write_event_record(
    event_id: str,
    ts: str,
    paths: dict,
    classifier: dict,
    latency_ms: dict,
    trigger_source: str,
    system_stats: dict,
    scheduler_context: Optional[dict],
) -> Path:
    record = {
        "event_id": event_id,
        "timestamp": ts,
        "trigger_source": trigger_source,
        "camera": "USB camera",
        "video_device": VIDEO_DEVICE,
        "paths": {
            "full": str(paths["full"]),
            "thumb": str(paths["thumb"]),
            "model_input": str(paths["model_input"]),
        },
        "classifier": {
            "label": classifier["label"],
            "confidence": classifier["confidence"],
            "top3": classifier["top3"],
            "model": MODEL_PATH.name,
            "input_dtype": str(inp["dtype"]),
            "output_dtype": str(out["dtype"]),
        },
        "latency_ms": latency_ms,
        "system": system_stats,
        "scheduler": scheduler_context,
    }

    event_json = EVENTS_DIR / f"{event_id}.json"
    event_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return event_json


# ---------------- PIR Scheduling ----------------
class PIRCaptureController:
    def __init__(
        self,
        cooldown_s: float = DEFAULT_PIR_COOLDOWN_S,
        followup_delay_s: float = DEFAULT_PIR_FOLLOWUP_DELAY_S,
        max_followups_per_episode: int = DEFAULT_PIR_MAX_FOLLOWUPS,
    ):
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.followup_delay_s = max(0.0, float(followup_delay_s))
        self.max_followups_per_episode = max(0, int(max_followups_per_episode))

        self._lock = threading.Lock()
        self._worker = None
        self._stop = threading.Event()

        self.motion_active = False
        self.motion_started_perf = None
        self.motion_started_iso = None
        self.motion_ended_perf = None
        self.motion_ended_iso = None
        self.episode_id = None

        self.cooldown_until_perf = 0.0
        self.pending_motion = False
        self.suppressed_trigger_count = 0
        self.followups_taken = 0
        self.last_accept_perf = None
        self.last_accept_iso = None

    def _worker_alive_locked(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _reset_episode_locked(self) -> None:
        self.motion_started_perf = None
        self.motion_started_iso = None
        self.motion_ended_perf = None
        self.motion_ended_iso = None
        self.episode_id = None
        self.pending_motion = False
        self.suppressed_trigger_count = 0
        self.followups_taken = 0
        self.last_accept_perf = None
        self.last_accept_iso = None

    def _complete_episode_locked(self, now_perf: float) -> None:
        if self.motion_started_perf is None:
            return

        observed_active_ms = None
        if self.motion_started_perf is not None and self.motion_ended_perf is not None:
            observed_active_ms = (self.motion_ended_perf - self.motion_started_perf) * 1000.0

        self.cooldown_until_perf = now_perf + self.cooldown_s
        print(
            f"[PIR] Episode {self.episode_id} complete. "
            f"active_ms={observed_active_ms if observed_active_ms is not None else 'open'} "
            f"cooldown={self.cooldown_s:.2f}s suppressed={self.suppressed_trigger_count}"
        )
        self._reset_episode_locked()

    def _build_scheduler_context_locked(self, trigger_source: str, trigger_perf: Optional[float]) -> dict:
        now_perf = time.perf_counter()
        pir_active_ms_observed = None
        if self.motion_started_perf is not None and self.motion_ended_perf is not None:
            pir_active_ms_observed = (self.motion_ended_perf - self.motion_started_perf) * 1000.0

        pir_active_ms_so_far = None
        if self.motion_started_perf is not None:
            end_perf = self.motion_ended_perf if self.motion_ended_perf is not None else now_perf
            pir_active_ms_so_far = (end_perf - self.motion_started_perf) * 1000.0

        scheduler_delay_ms = None
        if trigger_perf is not None:
            scheduler_delay_ms = (now_perf - trigger_perf) * 1000.0

        cooldown_remaining_ms = max(0.0, (self.cooldown_until_perf - now_perf) * 1000.0)

        return {
            "policy": {
                "mode": "pir_episode_scheduler",
                "cooldown_s": self.cooldown_s,
                "followup_delay_s": self.followup_delay_s,
                "max_followups_per_episode": self.max_followups_per_episode,
            },
            "episode": {
                "episode_id": self.episode_id,
                "motion_active": self.motion_active,
                "motion_started_at": self.motion_started_iso,
                "motion_ended_at": self.motion_ended_iso,
                "pir_active_ms_observed": pir_active_ms_observed,
                "pir_active_ms_so_far": pir_active_ms_so_far,
                "suppressed_trigger_count": self.suppressed_trigger_count,
                "followups_taken": self.followups_taken,
                "pending_motion": self.pending_motion,
            },
            "job": {
                "trigger_source": trigger_source,
                "accepted_at": self.last_accept_iso,
                "scheduler_delay_ms": scheduler_delay_ms,
                "cooldown_remaining_ms": cooldown_remaining_ms,
            },
        }

    def handle_motion(self) -> None:
        now_perf = time.perf_counter()
        with self._lock:
            if self.motion_started_perf is None and now_perf < self.cooldown_until_perf:
                self.suppressed_trigger_count += 1
                remaining_ms = (self.cooldown_until_perf - now_perf) * 1000.0
                print(f"[PIR] Ignored trigger during cooldown ({remaining_ms:.0f} ms remaining)")
                return

            self.motion_active = True

            if self.motion_started_perf is None:
                self.motion_started_perf = now_perf
                self.motion_started_iso = now_iso()
                self.motion_ended_perf = None
                self.motion_ended_iso = None
                self.episode_id = uuid.uuid4().hex[:8]
                self.pending_motion = False
                self.suppressed_trigger_count = 0
                self.followups_taken = 0
                print(f"[PIR] Motion episode started: {self.episode_id}")

            if self._worker_alive_locked():
                self.pending_motion = True
                self.suppressed_trigger_count += 1
                print("[PIR] Motion merged into current episode while pipeline is busy")
                return

            self.last_accept_perf = now_perf
            self.last_accept_iso = now_iso()
            self._worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker.start()
            print(f"[PIR] Accepted trigger for episode {self.episode_id}")

    def handle_no_motion(self) -> None:
        now_perf = time.perf_counter()
        with self._lock:
            if not self.motion_active and self.motion_started_perf is None:
                return

            self.motion_active = False
            self.motion_ended_perf = now_perf
            self.motion_ended_iso = now_iso()
            active_ms = None
            if self.motion_started_perf is not None:
                active_ms = (self.motion_ended_perf - self.motion_started_perf) * 1000.0
            print(f"[PIR] Motion cleared. active_ms={active_ms if active_ms is not None else 'n/a'}")

            if not self._worker_alive_locked():
                self._complete_episode_locked(now_perf)

    def _run_worker(self) -> None:
        trigger_source = "pir"
        trigger_perf = self.last_accept_perf

        while not self._stop.is_set():
            with self._lock:
                scheduler_context = self._build_scheduler_context_locked(trigger_source, trigger_perf)

            result = safe_handle_capture(trigger_source=trigger_source, scheduler_context=scheduler_context)
            if result is None:
                with self._lock:
                    now_perf = time.perf_counter()
                    if not self.motion_active:
                        self._complete_episode_locked(now_perf)
                return

            should_followup = False
            with self._lock:
                if (self.pending_motion or self.motion_active) and self.followups_taken < self.max_followups_per_episode:
                    self.pending_motion = False
                    self.followups_taken += 1
                    should_followup = True
                    next_followup_num = self.followups_taken
                    print(
                        f"[PIR] Scheduling follow-up capture {next_followup_num}/"
                        f"{self.max_followups_per_episode} for episode {self.episode_id}"
                    )
                else:
                    now_perf = time.perf_counter()
                    if not self.motion_active:
                        self._complete_episode_locked(now_perf)
                    else:
                        print(
                            f"[PIR] Waiting for motion to clear for episode {self.episode_id}; "
                            f"no extra capture scheduled"
                        )
                    return

            time.sleep(self.followup_delay_s)
            trigger_source = "pir_followup"
            trigger_perf = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()
        worker = None
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)


# ---------------- Orchestration ----------------
def handle_capture(trigger_source: str = "manual", scheduler_context: Optional[dict] = None) -> dict:
    with PIPELINE_LOCK:
        event_id = uuid.uuid4().hex[:12]
        ts = now_iso()
        event_folder = DATA_DIR / event_id
        event_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[ACTION] Triggered by {trigger_source}. Capturing (Event: {event_id})...")
        system_before = sample_system_stats()
        t0 = time.perf_counter()

        frame = capture_frame()
        t_cap = time.perf_counter()

        classifier = classify_frame(frame)
        t_inf = time.perf_counter()

        paths = save_event_artifacts(event_folder, frame, classifier["rgb_model_input"])
        t_save = time.perf_counter()

        status_code = None
        upload_error = None
        try:
            status_code = upload_to_pi5(ts, classifier["label"], classifier["confidence"], paths["thumb_img"])
            print(f"[NETWORK] POST to Pi 5: {status_code}")
        except Exception as e:
            upload_error = str(e)
            print(f"[NETWORK ERROR] Could not reach Pi 5: {e}")
        t_upload = time.perf_counter()

        latency_ms = {
            "capture": (t_cap - t0) * 1000,
            "inference": (t_inf - t_cap) * 1000,
            "save_artifacts": (t_save - t_inf) * 1000,
            "upload": (t_upload - t_save) * 1000,
            "end_to_end": (t_upload - t0) * 1000,
        }

        system_after = sample_system_stats()
        system_stats = {
            "before": system_before,
            "after": system_after,
            "delta": compute_system_delta(system_before, system_after),
        }

        event_json = write_event_record(
            event_id,
            ts,
            paths,
            classifier,
            latency_ms,
            trigger_source,
            system_stats,
            scheduler_context,
        )

        print(f"[AI RESULT] {classifier['label']} (Confidence: {classifier['confidence']:.3f})")
        print("Top-3:")
        for item in classifier["top3"]:
            print(f"  {item['label']:<16s} {item['score']:.3f}")
        print("Latency (ms):")
        for k, v in latency_ms.items():
            print(f"  {k:<16s} {v:8.2f}")
        if scheduler_context is not None:
            job_info = scheduler_context.get("job", {})
            print("Scheduler:")
            print(f"  delay_ms         {job_info.get('scheduler_delay_ms')}")
            print(f"  accepted_at      {job_info.get('accepted_at')}")
        print(f"[SAVED] {paths['full']}")
        print(f"[EVENT] {event_json}")

        return {
            "event_id": event_id,
            "timestamp": ts,
            "trigger_source": trigger_source,
            "classifier": {
                "label": classifier["label"],
                "confidence": classifier["confidence"],
                "top3": classifier["top3"],
            },
            "latency_ms": latency_ms,
            "system": system_stats,
            "scheduler": scheduler_context,
            "event_json": str(event_json),
            "upload_status_code": status_code,
            "upload_error": upload_error,
        }


def safe_handle_capture(trigger_source: str, scheduler_context: Optional[dict] = None) -> Optional[dict]:
    try:
        return handle_capture(trigger_source=trigger_source, scheduler_context=scheduler_context)
    except Exception as e:
        print(f"[ERROR] Capture pipeline failed: {e}")
        return None


# ---------------- Profile / Workload Modes ----------------
def run_manual_mode() -> None:
    print("System armed. Waiting for PIR, dashboard, or ENTER...")
    while True:
        input("Press ENTER to force a capture...\n")
        safe_handle_capture(trigger_source="manual")


def run_batch_mode(num_events: int, interval_s: float, sample_interval_s: float) -> Path:
    print(f"[mode] Batch mode: {num_events} event(s), {interval_s:.1f}s apart")
    run_id = uuid.uuid4().hex[:12]
    started_at = now_iso()
    sampler = SystemSampler(interval_s=sample_interval_s)
    events = []
    sampler.start()
    try:
        for i in range(num_events):
            print(f"\n--- Event {i + 1}/{num_events} ---")
            event_start = time.perf_counter()
            result = safe_handle_capture(trigger_source="batch")
            if result is not None:
                events.append(result)
            elapsed = time.perf_counter() - event_start
            sleep_time = max(0.0, interval_s - elapsed)

            if i < num_events - 1 and sleep_time > 0:
                print(f"[wait] Sleeping {sleep_time:.2f}s before next event")
                time.sleep(sleep_time)
    finally:
        sampler.stop()

    completed_at = now_iso()
    run_log_path = write_run_log(
        run_id=run_id,
        mode="batch",
        started_at=started_at,
        completed_at=completed_at,
        args={
            "batch": num_events,
            "interval": interval_s,
            "sample_interval": sample_interval_s,
        },
        events=events,
        sampler=sampler,
    )
    print(f"\n[done] Batch mode complete. Run log: {run_log_path}")
    return run_log_path


def run_timed_mode(duration_s: float, interval_s: float, sample_interval_s: float) -> Path:
    print(f"[mode] Timed mode: {duration_s:.1f}s total, {interval_s:.1f}s apart")
    run_id = uuid.uuid4().hex[:12]
    started_at = now_iso()
    sampler = SystemSampler(interval_s=sample_interval_s)
    events = []
    overall_start = time.perf_counter()
    event_count = 0
    sampler.start()
    try:
        while True:
            if time.perf_counter() - overall_start >= duration_s:
                break

            event_count += 1
            print(f"\n--- Event {event_count} ---")
            event_start = time.perf_counter()
            result = safe_handle_capture(trigger_source="timed")
            if result is not None:
                events.append(result)
            elapsed = time.perf_counter() - event_start
            sleep_time = max(0.0, interval_s - elapsed)

            if time.perf_counter() - overall_start >= duration_s:
                break

            if sleep_time > 0:
                print(f"[wait] Sleeping {sleep_time:.2f}s before next event")
                time.sleep(sleep_time)
    finally:
        sampler.stop()

    total_time = time.perf_counter() - overall_start
    completed_at = now_iso()
    run_log_path = write_run_log(
        run_id=run_id,
        mode="timed",
        started_at=started_at,
        completed_at=completed_at,
        args={
            "duration": duration_s,
            "interval": interval_s,
            "sample_interval": sample_interval_s,
            "elapsed_total": total_time,
        },
        events=events,
        sampler=sampler,
    )
    print(f"\n[done] Timed mode complete. Ran {event_count} event(s) in {total_time:.2f}s.")
    print(f"[done] Run log: {run_log_path}")
    return run_log_path


# ---------------- Triggers ----------------
def handle_pir_motion() -> None:
    if PIR_CONTROLLER is None:
        safe_handle_capture(trigger_source="pir")
    else:
        PIR_CONTROLLER.handle_motion()


def handle_pir_no_motion() -> None:
    if PIR_CONTROLLER is not None:
        PIR_CONTROLLER.handle_no_motion()


@sio.on("remote_capture")
def on_remote_trigger(data):
    threading.Thread(target=safe_handle_capture, args=("dashboard",), daemon=True).start()


# ---------------- Main ----------------
def main() -> None:
    global PIR_CONTROLLER

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="Run exactly N events")
    parser.add_argument("--duration", type=float, default=0, help="Run for N seconds")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between event starts")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between system resource samples in batch/timed mode")
    parser.add_argument(
        "--pir-cooldown",
        type=float,
        default=DEFAULT_PIR_COOLDOWN_S,
        help="Cooldown after a PIR motion episode ends",
    )
    parser.add_argument(
        "--pir-followup-delay",
        type=float,
        default=DEFAULT_PIR_FOLLOWUP_DELAY_S,
        help="Delay before an optional follow-up capture within the same motion episode",
    )
    parser.add_argument(
        "--pir-max-followups",
        type=int,
        default=DEFAULT_PIR_MAX_FOLLOWUPS,
        help="Maximum extra captures allowed within one PIR motion episode",
    )
    parser.add_argument(
        "--no-pir",
        action="store_true",
        help="Disable PIR trigger registration (useful for workload simulation)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable Socket.IO dashboard connection",
    )
    args = parser.parse_args()

    if args.batch > 0 and args.duration > 0:
        raise ValueError("Use either --batch or --duration, not both.")

    print(f"[ok] Loaded model: {MODEL_PATH.name}")
    print(f"[ok] Input dtype={inp['dtype']} shape={inp['shape']}")
    print(f"[ok] Output dtype={out['dtype']} shape={out['shape']}")

    pir_enabled = not args.no_pir and args.batch == 0 and args.duration == 0
    dashboard_enabled = CONNECT_DASHBOARD and (not args.no_dashboard)

    if pir_enabled:
        PIR_CONTROLLER = PIRCaptureController(
            cooldown_s=args.pir_cooldown,
            followup_delay_s=args.pir_followup_delay,
            max_followups_per_episode=args.pir_max_followups,
        )
        pir.when_motion = handle_pir_motion
        pir.when_no_motion = handle_pir_no_motion
        print("[ok] PIR trigger enabled")
        print(
            f"[ok] PIR scheduler: cooldown={args.pir_cooldown:.2f}s, "
            f"followup_delay={args.pir_followup_delay:.2f}s, "
            f"max_followups={args.pir_max_followups}"
        )
    else:
        print("[info] PIR trigger disabled")

    try:
        if dashboard_enabled:
            print(f"[info] Connecting to dashboard at {PI5_URL}...")
            sio.connect(PI5_URL)
        else:
            print("[info] Dashboard connection disabled")

        if args.batch > 0:
            run_batch_mode(args.batch, args.interval, args.sample_interval)
        elif args.duration > 0:
            run_timed_mode(args.duration, args.interval, args.sample_interval)
        else:
            run_manual_mode()

    except KeyboardInterrupt:
        print("\n[info] Exiting...")
    finally:
        pir.when_motion = None
        pir.when_no_motion = None
        if PIR_CONTROLLER is not None:
            PIR_CONTROLLER.stop()
        if sio.connected:
            sio.disconnect()
        HTTP.close()
        pir.close()


if __name__ == "__main__":
    main()
