# Edge Wildlife Classifier — Run Guide

This document describes how to run the edge image-capture + classification pipeline using:
- Raspberry Pi Zero 2 W (all-in-one inference node)

---

## 1. Physical Setup

### 1.1 Hardware
- Raspberry Pi Zero 2 W.
- Logitech C270 USB webcam.
- HC-SR501 PIR sensor.
- Power supply:
  - stable 5V supply.
  - via USB to Pi or a separate USB source.

### 1.2 Connections

Camera + inference node (Pi):

1. Plug the Logitech C270 into a USB port.

2. Confirm the camera enumerates:
   ```bash
   ls -l /dev/video*
   ```

## 2. Environment Setup

### 2.1 OS update + base packages
On the Pi (SSH recommended):
```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```
After reboot:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv
```

### 2.2 Create project folders

```bash
mkdir -p ~/wildlife_edge/{src,models,events,data}
```
### 2.3 Create a Python virtual environment
Create a venv that can access system packages (installed via apt):
```bash
cd ~/wildlife_edge
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```
Install Python dependencies and TFLite inference runtime:
```bash
pip install --upgrade pip
pip install numpy pillow
pip install tflite-runtime
```

## 3. Copying Model + Script to the Pi

### 3.1 Files Required
You need:
- `model_int8.tflite`
- `labels.json`
- `capture_and_classify.py` (pipeline script)

### 3.2 Copy files from Windows to Pi via SCP

From Windows Command Prompt (replace `<student@PI_IP>` if needed):
```bash
scp "%USERPROFILE%\Downloads\model_int8.tflite" student@<PI_IP>:~/wildlife_edge/models/
scp "%USERPROFILE%\Downloads\labels.json" student@<PI_IP>:~/wildlife_edge/models/
scp "%USERPROFILE%\Downloads\capture_and_classify.py" student@<PI_IP>:~/wildlife_edge/src/
```

## 4. Running the Application

### 4.1 Activate venv
```bash
cd ~/wildlife_edge
source .venv/bin/activate
```

### 4.2 Run the pipeline
```bash
python3 src/capture_and_classify.py
```

The script will:

1. Load the INT8 model + labels.
2. Wait for manual trigger (press ENTER).
3. Capture a single frame from the C270 (/dev/video0 by default).
4. Create a 224x224 input crop.
5. Run inference.
6. Save:
- full image: `~/wildlife_edge/data/<event_id>/full.jpg`
- thumbnail: `~/wildlife_edge/data/<event_id>/thumb.jpg`
- model input: `~/wildlife_edge/data/<event_id>/model_224.jpg`
- event JSON: `~/wildlife_edge/events/<event_id>.json`
7. Print top-3 predictions and latency breakdown.
