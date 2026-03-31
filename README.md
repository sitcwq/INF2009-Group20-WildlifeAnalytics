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
- `edge_with_logs_scheduled.py` (pipeline script)

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
python3 src/edge_with_logs_scheduled.py [options]
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

## 5. Available Arguments / Options

### 5.1 `--batch N`
Run exactly `N` capture events, then exit.
- Type: `int`
- Default: `0`

Example:
```bash
python3 edge_with_logs_scheduled.py --batch 10
```

### 5.2 `--duration SECONDS`
Run the timed workload mode for a total of `SECONDS`.
- Type: `float`
- Default: `0`

Example:
```bash
python3 edge_with_logs_scheduled.py --duration 600
```

### 5.3 `--interval SECONDS`
Seconds between event starts in batch or timed mode.
- Type: `float`
- Default: `10.0`

Example:
```bash
python3 edge_with_logs_scheduled.py --duration 600 --interval 5
```

### 5.4 `--sample-interval SECONDS`
Sampling interval for system resource logging in batch or timed mode.
- Type: `float`
- Default: `1.0`

Example:
```bash
python3 edge_with_logs_scheduled.py --batch 20 --sample-interval 0.5
```

### 5.5 `--pir-cooldown SECONDS`
Cooldown after a PIR motion episode ends.
- Type: `float`
- Default: `2.0`

Example:
```bash
python3 edge_with_logs_scheduled.py --pir-cooldown 3
```
### 5.6 `--pir-followup-delay SECONDS`
Delay before an optional follow-up capture within the same PIR motion episode.
- Type: `float`
- Default: `0.5`

Example:
```bash
python3 edge_with_logs_scheduled.py --pir-followup-delay 1.0
```

### 5.7 `--pir-max-followups N`
Delay before an optional follow-up capture within the same PIR motion episode.
- Type: `int`
- Default: `1`

Example:
```bash
python3 edge_with_logs_scheduled.py --pir-max-followups 2
```
### 5.8 `--no-pir`
Disable PIR trigger registration.
- Type: flag
- Default: disabled unless explicitly provided

Example:
```bash
python3 edge_with_logs_scheduled.py --duration 600 --no-pir
```
### 5.9 `--no-dashboard`
Disable Socket.IO dashboard connection.
- Type: flag
- Default: disabled unless explicitly provided

Example:
```bash
python3 edge_with_logs_scheduled.py --duration 600 --no-dashboard
```
### 5.10 `--no-dashboard`
Disable Socket.IO dashboard connection.
- Type: flag
- Default: disabled unless explicitly provided

Example:
```bash
python3 edge_with_logs_scheduled.py --duration 600 --no-dashboard
```

### Important Rules

You cannot use both:
- `--batch`
- `--duration`

at the same time. This will raise an error.

### Notes

- If neither `--batch` nor `--duration` is given, the script runs in manual/PIR mode.
- `--interval` and `--sample-interval` matter primarily in batch and timed modes.
- `--no-pir` is useful for workload simulation without motion-trigger input.
- `--no-dashboard` prevents connection to the remote dashboard server.

## 6. Other Python Files in this Repository

There are two other files/scripts used extensively in this project:

- `download_inat_sg.py`
- `summarize_run.py`

### 6.1 `download_inat_sg.py`
This script is a dataset collection utility for downloading wildlife training images from [iNaturalist](https://www.inaturalist.org/) via the iNat observations API. It targets a fixed set of taxa, prioritizes Singapore first, then nearby fallback place IDs, downloads one photo per observation into class folders, and records everything in a manifest.csv so duplicate observation/photo pairs are not re-downloaded across runs. It also supports filtering to specific labels, selecting photo size, controlling retry behavior, and tuning API pacing.

### 6.2 `summarize_run.py`
This script is a post-run analysis utility for a JSON log produced by the edge pipeline. It loads the run log, identifies the first and last pipeline events, computes average/min/max latency for each pipeline stage, and summarizes CPU and RAM usage from the before/after system snapshots embedded in each event.

## 7. Team Members and Contributions

### Chia Wenqi

- Development of quantized MobileNetV3Small wildlife inference model
- Performance testing of edge pipeline (power, latency, and resource utilisation)
- Video demo

### Ong Hong Liang

- PPT slides
- Latency testing
- Video recording

### Jordan Chan Jun Xiang

- Dashboard integration
- PIR sensor setup and integration
- Video recording

### Jarrett Yeo Chung Yao

- Developement of initial YoloV8n wildlife inference model
- PPT slides
- Video recording

### Andrea Lye Wei Xuan

- Development of Dashboard
- Poster
- PPT slides
- Video recording

## 8. Acknowledgements
The team would like to thank Prof. Muhamed Fauzi for his continuous feedback and guidance throughout the development of this project. We would also like to thank Prof. Haja Rowther for suggesting useful tools and modules, and for loaning the equipment used in this project.

## 9. Declaration of Use of Generative AI Tools
Declaration of Use of Generative AI Tools
The team used generative AI tools, including Claude, ChatGPT and Gemini as supporting tools during selected stages of the project.

These tools were used mainly for the following purposes:

- Linux/Pi commands for troubleshooting
- Generation of scripts (`download_inat_sg.py` and `summarize_run.py`)
- Suggestions for improving the inference model

The generative AI tools were used as aids for development and for optimising the performance of the project. Final design choices, implementation decisions, experiments, analysis, and written conclusions remained the responsibility of the team. AI-assisted outputs were used selectively as supporting inputs during development and report preparation.
