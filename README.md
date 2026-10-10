# 🛡️ Roots Industrial PPE Compliance Detector

An enterprise-grade, real-time Computer Vision system engineered for industrial factory CCTV cameras, NVR surveillance systems, and live webcams. Automatically detects and logs Personal Protective Equipment (PPE) compliance across multiple workers simultaneously with sub-frame latency and high-precision spatial localization.

---

## 📌 Key Capabilities

- **Real-Time Multi-Worker Tracking**: Leverages fine-tuned YOLOv8m + YOLO11n coupled with ByteTrack to track multiple workers across cluttered industrial environments.
- **Persistent Worker Registry**: Re-identifies workers and locks transient tracker IDs to permanent worker identifiers (`Worker #1`, `Worker #2`), preventing tracker churn.
- **Ceiling & Equipment False-Positive Suppression**: Advanced spatial physics and geometric aspect ratio filters eliminate false detections on ceiling fans, hanging lights, ductwork, and vibrating machinery.
- **Lower-Face Mask & Respirator Alignment**: Specifically localizes mouth and nose protection, detecting both **3M P100 pink/magenta cartridges** and **surgical blue face masks** while rejecting hair/forehead false positives.
- **Anatomical Palm & Glove Inspection**: Isolates the lower arm, wrist, and palm region (`ph * 0.44` to `0.90`) to accurately detect gloves and bare skin without false alarms triggered by short sleeves or bare biceps.
- **Dual Operational Modes**:
  - 🌐 **Modern Web Dashboard**: Interactive React + Vite interface with live MJPEG streaming, compliance metrics, violation event timeline, and CSV export.
  - 🖥️ **High-Performance Terminal / Desktop GUI**: Ultra-low-latency OpenCV engine with colored terminal logging and heads-up overlay.

---

## 🏗️ System Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │         Industrial CCTV / NVR Network        │
                 │      (RTSP / H.264 / H.265 Sub-Streams)      │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │       Contrast Enhancement & CLAHE           │
                 │     (Low-Light / Spray Booth Optimization)   │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │   Multi-Person Detection (YOLO11n + ByteTrack)│
                 └──────────────────────┬───────────────────────┘
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│  Ceiling & Equipment Filter   │               │   Persistent Worker Registry  │
│  - Rejects by2 < 0.42h        │               │   - Locks transient ByteTrack │
│  - Rejects wall fan sector    │               │     IDs to Worker #1, #2      │
│  - Enforces aspect ratio      │               │   - Multi-frame gear memory   │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │  Smart Head Inspection │               │ Smart Hand Inspection │
        │  - 3M P100 pink discs │               │ - Lower palm zones    │
        │  - Blue surgical mask │               │ - Bare skin detection │
        │  - Lower face clamp   │               │ - Short sleeve filter │
        └───────────┬───────────┘               └───────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                 Compliance Decision Engine                    │
        │           (COMPLIANT / NON-COMPLIANT + Event Log)             │
        └───────────────┬───────────────────────────────┬───────────────┘
                        │                               │
                        ▼                               ▼
        ┌───────────────────────────────┐ ┌───────────────────────────────┐
        │   FastAPI Web Stream Engine   │ │  Desktop OpenCV / Terminal    │
        │     (Port 8000 / MJPEG)       │ │      (Live Video Canvas)      │
        └───────────────────────────────┘ └───────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Node.js**: v18+ (for Web Dashboard)
- **NVIDIA GPU** (Recommended): CUDA 11.8+ or 12.x for RTX/GTX acceleration (CPU mode supported automatically)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ethi25/roots-ppe-detection.git
cd roots-ppe-detection

# Install Python requirements
pip install ultralytics opencv-python numpy fastapi uvicorn pydantic requests

# Install Frontend requirements
cd frontend
npm install
cd ..
```

---

## 🔒 Security Configuration (Best Practices)

### Step 1: Create Your `.env` File
Do **not** commit real surveillance credentials to GitHub. Copy the provided `.env.example` file:

```bash
copy .env.example .env
```

Edit `.env` with your secure credentials:

```ini
# CCTV NVR / IP Camera Network Settings
CCTV_NVR_IP=192.168.127.5
CCTV_NVR_PORT=554
CCTV_NVR_USER=admin
CCTV_NVR_PASS=YourSecurePasswordHere
CCTV_BRAND=hikvision

# Default Camera Channel
DEFAULT_CAMERA_CHANNEL=3

# Web Application Host
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

### Step 2: Surveillance Network Security Guidelines

1. **Dedicated Read-Only CCTV Account**:
   - Create a dedicated user account on your NVR (e.g., `ppe_reader`) with **Read-Only / Live View** privileges instead of using the master `admin` account.
2. **Surveillance VLAN Segmentation**:
   - Keep industrial CCTV cameras on an isolated physical network or VLAN (e.g., `192.168.127.0/24`) inaccessible from the public internet.
3. **Web Dashboard Access**:
   - By default, the web dashboard binds to `127.0.0.1` (localhost only).
   - If deploying to a company intranet, place the backend behind a reverse proxy (e.g., **Nginx** or **Caddy**) with HTTPS/TLS encryption and basic or JWT authentication.
4. **Credential Isolation**:
   - The `.gitignore` file is pre-configured to strictly ignore `.env`, `.env.local`, video recordings, and local test logs.

---

## 🖥️ Running the Application

### Option A: Web Dashboard (Recommended)

To launch the full web interface (FastAPI backend + Vite React frontend):

**Windows (One-Click):**
Double-click `run_webapp.bat`.

**Manual Terminal Start:**
```bash
# Terminal 1 - Start Backend API
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --http h11 --ws none

# Terminal 2 - Start Frontend
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open your browser to: **`http://localhost:5173`**

From the top camera selector dropdown, seamlessly toggle between:
- `[D3] PC Spray Booth West Wall` (Ch 302)
- `[D29] Powder Coating Office Area` (Ch 2902)
- `[D23] Tool Room` (Ch 2302)
- `Live Webcam` (Camera 0)

---

### Option B: Standalone Desktop / Terminal Engine

To run detection directly with an OpenCV GUI preview and terminal logs:

**Windows (One-Click Launchers):**
- `run_cctv_menu.bat`: Interactive menu to select camera channel.
- `run_cctv_d3.bat`: Directly monitors **[D3] PC Spray Booth West Wall**.
- `run_cctv_d29.bat`: Directly monitors **[D29] Powder Coating Office Area**.
- `run_webcam.bat`: Runs on local laptop/USB webcam.

**Command-Line Options:**
```bash
# Run on Spray Booth Camera (Channel 3)
python terminal_detector.py --channel 3 --area "PC SPRAY BOOTH WEST WALL"

# Run on Office Area Camera (Channel 29)
python terminal_detector.py --channel 29 --area "POWDER COATING OFFICE AREA"

# Run on Custom RTSP URL
python terminal_detector.py --video "rtsp://admin:password@192.168.1.100:554/ch1/main"

# Run Headless (no GUI window, high-speed server logging)
python terminal_detector.py --channel 3 --headless
```

---

## 🧠 Detection & Filtering Logic

### 1. Ceiling & Fan Elimination Filter
Industrial wall fans and ceiling mounts frequently fool generic person detectors. This system enforces 4 layers of rejection:
1. **Vertical Footing Guard (`by2 < 0.42 * h`)**: Grounded workers stand on the floor; suspended objects ending in the upper 40% are discarded.
2. **Ceiling Attachment Check (`by1 < 0.12 * h` and `by2 < 0.60 * h`)**: Identifies top-hung light fixtures, ducts, and overhead mounts.
3. **Sector Exclusion**: Blocks fixed equipment regions in overhead booths.
4. **Human Aspect Ratio**: Rejects square or wide objects (`ar < 0.85`).

### 2. Anatomical PPE Localization
- **Respirator & Mask Detection**:
  - 3M P100 pink cartridge filters are segmented using dual-range HSV combined with chromatic guard (`R > 110, B > 60, R > G + 18`).
  - Surgical masks are recognized by trained YOLOv8m weights on the lower half of the head crop (`0.45 * hh` to `0.90 * hh`).
- **Gloves & Exposed Hands**:
  - Inspection zones are placed specifically at the wrists and hands (`0.44 * ph` to `0.90 * ph`), preventing short-sleeve false alarms.
  - Exposed palm skin triggers `NO GLOVE` alerts with boxes placed directly on the worker's hands.

---

## 📁 Repository Structure

```
├── backend/
│   ├── main.py                  # FastAPI REST endpoints & live video feed router
│   ├── streamer.py              # Real-time multi-threaded CCTV pipeline & AI engine
│   ├── utils.py                 # Color filtering, CLAHE pre-processing & geometry math
│   ├── yolov8m-ppe.pt           # Fine-tuned industrial PPE model weights
│   ├── yolo11n.pt               # Ultralytics human person tracker weights
│   └── violations_log.csv       # Persistent event log for safety compliance audits
├── frontend/
│   ├── src/                     # React application source code
│   │   ├── App.jsx              # Main dashboard component
│   │   └── index.css            # Dark-mode industrial design system
│   └── vite.config.js           # Vite development and proxy configuration
├── bytetrack_factory.yaml       # Tuned multi-worker ByteTrack configuration
├── terminal_detector.py         # Standalone desktop CLI & OpenCV inspection engine
├── run_webapp.bat               # One-click launcher for Web Dashboard
├── run_cctv_menu.bat            # Interactive launcher menu for CCTV channels
├── .env.example                 # Template for secure NVR credential management
└── .gitignore                   # Excludes sensitive credentials, logs, and video caches
```

---

## 📄 License & Attribution

Designed and engineered for **Roots Industries** industrial safety monitoring. Built using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [FastAPI](https://fastapi.tiangolo.com/), and [React](https://react.dev/).
