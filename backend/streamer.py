"""
ROOTS PPE WEB STREAMER
Hardware-accelerated live CCTV streamer and stats provider for the web app.
Uses the fine-tuned yolov8m-ppe.pt model and RTX 4050 GPU.
"""

import os
import sys
import time
import re
import cv2
import numpy as np
import torch
from collections import deque
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.dirname(BASE_DIR)
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8m-ppe.pt")
PERSON_WEIGHTS_PATH = os.path.join(SCRATCH_DIR, "yolo11n.pt")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
CSV_LOG_PATH = os.path.join(BASE_DIR, "violations_log.csv")

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

PPE_CLASSES = {
    0: 'glove',
    1: 'goggles',
    2: 'helmet',
    3: 'mask',
    4: 'no_glove',
    5: 'no_goggles',
    6: 'no_helmet',
    7: 'no_mask',
    8: 'no_shoes',
    9: 'shoes'
}

COLOR_PALETTE = {
    'mask': (255, 200, 0),        # Cyan/Sky blue
    'respirator': (255, 200, 0),
    'glove': (0, 220, 255),       # Amber yellow
    'helmet': (255, 180, 0),
    'shoes': (255, 0, 255),
    'goggles': (255, 255, 0),
    'no_mask': (0, 0, 240),       # Red
    'no_glove': (0, 0, 240),      # Red
    'no_helmet': (0, 0, 240),
    'no_shoes': (0, 0, 240),
}

# --- Shared Live Stream State ---
class StreamState:
    def __init__(self):
        self.camera_name = "[D29] POWDER COATING OFFICE AREA"
        self.camera_key = "d29"
        self.is_running = False
        self.fps = 0.0
        self.active_workers = 0
        self.compliant_workers = 0
        self.violating_workers = 0
        self.compliance_rate = 100
        self.device_name = "CPU"
        self.recent_events = deque(maxlen=30)
        self.current_frame_jpeg = None
        self.last_frame_time = time.time()
        self.lock = None

stream_state = StreamState()

class PersistentWorkerRegistry:
    """
    Industrial Persistent Worker Registry.
    Bridges transient ByteTrack IDs (which jump to 126, 127 etc. after brief occlusions)
    to permanent, human-friendly worker IDs (Worker #1, Worker #2, etc.)
    using spatial proximity and motion continuity.
    """
    def __init__(self, max_idle_time=45.0, max_match_dist_norm=0.20):
        self.max_idle_time = max_idle_time
        self.max_match_dist_norm = max_match_dist_norm
        self.slots = {}
        self.tid_to_slot = {}

    def update(self, detected_boxes, byte_track_ids, current_time, frame_w, frame_h):
        dead_slots = [
            slot for slot, data in self.slots.items()
            if (current_time - data['last_seen']) > self.max_idle_time
        ]
        for slot in dead_slots:
            del self.slots[slot]
            self.tid_to_slot = {t: s for t, s in self.tid_to_slot.items() if s != slot}

        diag = np.sqrt(frame_w**2 + frame_h**2)
        max_dist_px = self.max_match_dist_norm * diag

        resolved_slots = [None] * len(detected_boxes)
        unmatched_indices = []

        # 1. Match known ByteTrack IDs first
        for i, (box, tid) in enumerate(zip(detected_boxes, byte_track_ids)):
            if tid in self.tid_to_slot and self.tid_to_slot[tid] in self.slots:
                slot = self.tid_to_slot[tid]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                self.slots[slot]['last_seen'] = current_time
                self.slots[slot]['box'] = box
                self.slots[slot]['centroid'] = (cx, cy)
                self.slots[slot]['last_tid'] = tid
                resolved_slots[i] = slot
            else:
                unmatched_indices.append(i)

        available_slots = [slot for slot in self.slots if slot not in resolved_slots]

        # 2. For newly spawned ByteTrack IDs (e.g. 126), re-identify using nearest idle slot
        for i in unmatched_indices:
            box = detected_boxes[i]
            tid = byte_track_ids[i]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            best_slot = None
            best_dist = float('inf')

            for slot in available_slots:
                prev_cx, prev_cy = self.slots[slot]['centroid']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if dist < max_dist_px and dist < best_dist:
                    best_dist = dist
                    best_slot = slot

            if best_slot is not None:
                self.tid_to_slot[tid] = best_slot
                self.slots[best_slot]['last_seen'] = current_time
                self.slots[best_slot]['box'] = box
                self.slots[best_slot]['centroid'] = (cx, cy)
                self.slots[best_slot]['last_tid'] = tid
                resolved_slots[i] = best_slot
                available_slots.remove(best_slot)
            else:
                assigned_slot = 1
                while assigned_slot in self.slots:
                    assigned_slot += 1
                self.slots[assigned_slot] = {
                    'last_seen': current_time,
                    'box': box,
                    'centroid': (cx, cy),
                    'last_tid': tid
                }
                self.tid_to_slot[tid] = assigned_slot
                resolved_slots[i] = assigned_slot

        return resolved_slots

def denoise_frame(frame, clahe_clip=1.8):
    """
    Apply CLAHE in LAB color space to equalize lamp glare and sharpen worker/gear edges in dark CCTV areas.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# --- Detectors ---
def detect_industrial_cartridge_in_lower_face(head_crop):
    """
    Detects 3M industrial cartridge respirators (P100 magenta/pink cartridges)
    located strictly in the lower face (nose/mouth/cheeks: rows 28% to 94% of head).
    NOTE: Yellow cartridge detection DISABLED — only pink and blue masks are valid.
    Returns: (has_cartridge, bounding_box, confidence)
    """
    if head_crop is None or head_crop.size == 0:
        return False, None, 0.0

    hh, hw = head_crop.shape[:2]
    # Lower face region (mouth, nose, and cheek cartridge zones)
    y1, y2 = int(hh * 0.28), int(hh * 0.94)
    x1, x2 = int(hw * 0.05), int(hw * 0.95)
    lower_face = head_crop[y1:y2, x1:x2]
    if lower_face.size == 0:
        return False, None, 0.0

    blurred = cv2.GaussianBlur(lower_face, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1. 3M Pink/Magenta P100 cartridges:
    # Pure magenta/fuchsia filter disc.
    # Uses dual-range HSV (146-176) combined with BGR chromatic guard (R > 110, B > 60, R > G+18)
    # Excludes human skin/lips, orange reflections, and dark hair.
    pink_hsv = cv2.inRange(hsv, np.array([146, 42, 50]), np.array([176, 255, 255]))
    
    b_ch = lower_face[:, :, 0].astype(int)
    g_ch = lower_face[:, :, 1].astype(int)
    r_ch = lower_face[:, :, 2].astype(int)
    pink_bgr = ((r_ch > 110) & (b_ch > 60) & (r_ch > g_ch + 18) & (b_ch > g_ch + 5)).astype(np.uint8) * 255
    
    pink = cv2.bitwise_and(pink_hsv, pink_bgr)

    # NOTE: Yellow cartridge detection removed — yellow plates/railings caused false positives.
    # Only pink (P100) respirators and blue surgical masks are valid.

    cartridges = pink
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cartridges = cv2.morphologyEx(cartridges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cartridges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    max_score = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        # Min contour area: for distant workers 20px, for close workers 100+px
        if area >= 20:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Circular or oval cartridge shape: aspect ratio 0.40 to 2.5
            ar = cw / float(max(1, ch))
            if 0.40 <= ar <= 2.5:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(1, hull_area)
                if solidity >= 0.48:
                    score = min(0.98, 0.72 + (area / 800.0) * 0.24)
                    if score > max_score:
                        max_score = score
                        best_box = [x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch]

    if best_box is not None:
        return True, best_box, max_score
    return False, None, 0.0


def detect_bare_hands_in_zone(zone_crop):
    """Detects exposed bare skin in hand/palm crops. Returns (has_bare_skin, skin_ratio, palm_box)."""
    if zone_crop is None or zone_crop.size == 0:
        return False, 0.0, None

    zz = cv2.GaussianBlur(zone_crop, (3, 3), 0)
    hsv = cv2.cvtColor(zz, cv2.COLOR_BGR2HSV)
    skin_lo  = cv2.inRange(hsv, np.array([  0, 18,  70]), np.array([ 22, 170, 255]))
    skin_hi  = cv2.inRange(hsv, np.array([165, 18,  70]), np.array([179, 170, 255]))
    skin = cv2.bitwise_or(skin_lo, skin_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)

    total_px = zone_crop.shape[0] * zone_crop.shape[1]
    if total_px == 0:
        return False, 0.0, None
    skin_ratio = np.count_nonzero(skin) / float(total_px)

    palm_box = None
    if skin_ratio > 0.08:
        contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 30:
                cx, cy, cw, ch = cv2.boundingRect(largest)
                palm_box = [cx, cy, cx + cw, cy + ch]

    return (skin_ratio > 0.08 and skin_ratio < 0.60), skin_ratio, palm_box


def draw_label(img, text, pt, bg_color, text_color=(255, 255, 255), scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pt
    x = max(5, x)
    y = max(th + 10, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


# --- Camera Resolution & Secure Environment Configuration ---
def load_env_file():
    for env_path in [".env", os.path.join("..", ".env"), os.path.join(os.path.dirname(__file__), "..", ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass

load_env_file()

DEFAULT_NVR_IP = os.getenv("CCTV_NVR_IP", "192.168.127.5")
DEFAULT_NVR_USER = os.getenv("CCTV_NVR_USER", "admin")
DEFAULT_NVR_PASS = os.getenv("CCTV_NVR_PASS", "dmin@123")
DEFAULT_NVR_PORT = os.getenv("CCTV_NVR_PORT", "554")

PRESET_CAMERAS = {
    "d29": {
        "name": "[D29] POWDER COATING OFFICE AREA",
        "ip": DEFAULT_NVR_IP,
        "user": DEFAULT_NVR_USER,
        "pass": DEFAULT_NVR_PASS,
        "port": DEFAULT_NVR_PORT,
        "channel": 29
    },
    "d3": {
        "name": "[D3] PC SPRAY BOOTH WEST WALL",
        "ip": DEFAULT_NVR_IP,
        "user": DEFAULT_NVR_USER,
        "pass": DEFAULT_NVR_PASS,
        "port": DEFAULT_NVR_PORT,
        "channel": 3
    },
    "d23": {
        "name": "[D23] TOOL ROOM",
        "ip": DEFAULT_NVR_IP,
        "user": DEFAULT_NVR_USER,
        "pass": DEFAULT_NVR_PASS,
        "port": DEFAULT_NVR_PORT,
        "channel": 23
    },
    "webcam": {
        "name": "Live Webcam",
        "source": 0
    },
    "demo": {
        "name": "Extracted Factory Footage (Demo)",
        "source": "demo"
    }
}

def resolve_source_stream(key="d29"):
    cfg = PRESET_CAMERAS.get(key.lower(), PRESET_CAMERAS["d29"])
    if "source" in cfg:
        return cfg["source"], cfg["name"]

    ip = cfg["ip"]
    user = cfg["user"]
    pwd = cfg["pass"]
    port = cfg.get("port", DEFAULT_NVR_PORT)
    ch = cfg["channel"]
    auth = f"{user}:{pwd}@"
    # Candidates in order
    candidates = [
        f"rtsp://{auth}{ip}:{port}/Streaming/Channels/{ch}02",  # Sub-stream
        f"rtsp://{auth}{ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",
        f"rtsp://{auth}{ip}:{port}/Streaming/Channels/{ch}01",  # Main-stream
        f"rtsp://{auth}{ip}:{port}/live/ch{ch}",
    ]
    return candidates, cfg["name"]


# --- Background Worker Manager ---
class VideoPipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        stream_state.device_name = self.dev_name
        
        # Load models and tracker
        print(f"[*] Initializing VideoPipeline on {self.dev_name}...")
        self.ppe_model = YOLO(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov8m-ppe.pt")
        self.person_model = YOLO(PERSON_WEIGHTS_PATH if os.path.exists(PERSON_WEIGHTS_PATH) else "yolo11n.pt")
        
        tracker_path = os.path.join(SCRATCH_DIR, "bytetrack_factory.yaml")
        self.tracker = tracker_path if os.path.exists(tracker_path) else "bytetrack.yaml"
        print(f"[*] Using Tracker: {os.path.basename(self.tracker)}")
        self.worker_registry = PersistentWorkerRegistry()
        self.worker_gear_memory = {}

        # Warmup
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.person_model.track(dummy, persist=True, classes=[0], imgsz=320, tracker=self.tracker, verbose=False, device=self.device)
        self.ppe_model.predict(dummy, conf=0.10, imgsz=320, verbose=False, device=self.device)
        print("[*] Models warmed up on GPU/CPU.")

    def generate_feed(self, camera_key="d29", stride=2):
        stream_state.camera_key = camera_key
        source_target, cam_name = resolve_source_stream(camera_key)
        stream_state.camera_name = cam_name
        self.worker_registry = PersistentWorkerRegistry()

        cap = None
        is_demo = False

        if source_target == "demo":
            is_demo = True
            # Load extracted frames slideshow
            frames_dir = os.path.join(BASE_DIR, "extracted_frames")
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
            if not frame_files:
                is_demo = False
                source_target = 0
            else:
                demo_idx = 0
        else:
            is_demo = False
            # Check if list of RTSP candidates
            if isinstance(source_target, list):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2500000"
                opened = False
                # First check socket reachability to avoid blocking
                cfg = PRESET_CAMERAS.get(camera_key.lower(), {})
                target_ip = cfg.get("ip", "192.168.127.5")
                reachable = False
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.8)
                    if sock.connect_ex((target_ip, 554)) == 0:
                        reachable = True
                    sock.close()
                except Exception:
                    reachable = False

                if reachable:
                    for cand in source_target:
                        try:
                            test_cap = cv2.VideoCapture(cand, cv2.CAP_FFMPEG)
                            if test_cap.isOpened():
                                ret, _ = test_cap.read()
                                if ret:
                                    cap = test_cap
                                    opened = True
                                    print(f"[*] Connected to RTSP stream: {cand}")
                                    break
                            test_cap.release()
                        except Exception as e:
                            print(f"[!] RTSP test error: {e}")
                else:
                    print(f"[*] Camera host {target_ip}:554 unreachable. Switching directly to demo mode.")

                if not opened:
                    print(f"[!] Could not connect to RTSP stream for {cam_name}, falling back to demo mode.")
                    is_demo = True
                    frames_dir = os.path.join(BASE_DIR, "extracted_frames")
                    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
                    demo_idx = 0
            else:
                cap = cv2.VideoCapture(source_target)

        fps_tracker = deque(maxlen=20)
        worker_cache = {}
        frame_idx = 0
        try:
            while True:
                t0 = time.time()
                if is_demo:
                    frame_path = frame_files[demo_idx % len(frame_files)]
                    frame = cv2.imread(frame_path)
                    demo_idx += 1
                    time.sleep(0.04) # ~25 FPS
                else:
                    if cap is None or not cap.isOpened():
                        break
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                frame_idx += 1
                h, w = frame.shape[:2]

                orig_h, orig_w = frame.shape[:2]
                # Ensure minimum 1280px resolution for detection
                min_w = 1280
                if orig_w < min_w:
                    up_scale = min_w / orig_w
                    frame = cv2.resize(frame, (min_w, int(orig_h * up_scale)), interpolation=cv2.INTER_LINEAR)
                    h, w = frame.shape[:2]
                else:
                    h, w = orig_h, orig_w

                active_workers_count = 0
                compliant_count = 0
                violating_count = 0
    
                # Run detection every 'stride' frames or re-use cache
                if frame_idx % stride == 0 or not worker_cache:
                    # Dynamically use 1536px for 4K / 2K / 1080p CCTV footage to keep distant workers clearly detected
                    imgsz_track = 1536 if (w >= 1920 or h >= 1080) else 1280
                    frame_enhanced = denoise_frame(frame, clahe_clip=1.8)
                    track_res = self.person_model.track(
                        frame_enhanced, persist=True, classes=[0], conf=0.15, imgsz=imgsz_track,
                        tracker=self.tracker, verbose=False, device=self.device
                    )
    
                    current_cache = {}
                    if track_res and len(track_res) > 0 and track_res[0].boxes and len(track_res[0].boxes) > 0:
                        raw_boxes = track_res[0].boxes.xyxy.cpu().numpy()
                        raw_confs = track_res[0].boxes.conf.cpu().numpy()
                        raw_ids = track_res[0].boxes.id.int().cpu().numpy() if track_res[0].boxes.id is not None else list(range(1, len(raw_boxes) + 1))

                        # ── STRICT HUMAN-WORKER VERIFICATION FILTER ──
                        # Excludes ceiling fans, hanging ductwork, lights, wall brackets, and machines.
                        # Real workers stand on the floor or sit at worktables; their bottom reaches the lower half of the frame.
                        valid_indices = []
                        for bi, (bx, conf) in enumerate(zip(raw_boxes, raw_confs)):
                            bx1, by1, bx2, by2 = bx
                            bw = max(1, bx2 - bx1)
                            bh = max(1, by2 - by1)
                            ar = bh / float(bw)
                            area = bw * bh
                            cy = (by1 + by2) / 2.0
                            cx = (bx1 + bx2) / 2.0
                            cy_norm = cy / float(h)
                            cx_norm = cx / float(w)
                            by1_norm = by1 / float(h)
                            by2_norm = by2 / float(h)

                            # Filter 1: Low confidence = noise/equipment reflection
                            if conf < 0.12:
                                continue
                            # Filter 2: Ceiling suspended object filter (fans, hanging lamps, ducts)
                            # Humans stand on the floor; a person NEVER floats entirely in the top 42% of the frame
                            if by2_norm < 0.42:
                                continue
                            # Filter 3: Top-hung fixture (starts in top 12% of frame and doesn't reach floor)
                            if by1_norm < 0.12 and by2_norm < 0.60:
                                continue
                            # Filter 4: Wall fan exclusion zone (Spray Booth West Wall fan at x: 35-62%, y: 5-42%)
                            if 0.35 <= cx_norm <= 0.62 and cy_norm <= 0.42:
                                continue
                            # Filter 5: Non-human aspect ratio (fans, plates, square machines)
                            if ar < 0.85:
                                continue
                            # Filter 6: Upper frame squat objects
                            if cy_norm < 0.50 and ar < 1.35:
                                continue
                            # Filter 7: Minimum physical size
                            if bh < 60 or area < 1500:
                                continue

                            valid_indices.append(bi)

                        if not valid_indices:
                            worker_cache = current_cache
                            continue

                        boxes = raw_boxes[valid_indices]
                        track_ids = [raw_ids[i] for i in valid_indices]
                        
                        # Persistent Worker Registry: Locks transient ByteTrack IDs (e.g. 126) to permanent Worker #1, #2
                        worker_slot_ids = self.worker_registry.update(boxes, track_ids, time.time(), w, h)

                        for box, tid in zip(boxes, worker_slot_ids):
                            active_workers_count += 1
                            px1, py1, px2, py2 = map(int, box)
                            pw = max(1, px2 - px1)
                            ph = max(1, py2 - py1)
    
                            # ── 1. Smart Head Zoom Inspection (Mask / No Mask) ──
                            # Robust upper-body vs full-body detection
                            aspect = ph / float(max(1, pw))
                            is_upper_body = (aspect < 1.35) and (ph < 350)
                            head_down_ratio = 0.52 if is_upper_body else 0.32

                            # Overhead CCTV perspective compensation: expand upward to include head
                            hy1 = max(0, py1 - int(ph * 0.16))
                            hy2 = min(h, py1 + int(ph * head_down_ratio))
                            hx1 = max(0, px1 - int(pw * 0.08))
                            hx2 = min(w, px2 + int(pw * 0.08))
                            head_crop = frame[hy1:hy2, hx1:hx2]
                            
                            scale = 1.0
                            has_mask = False
                            mask_box_global = None
                            
                            if head_crop.size > 0:
                                _hh, _hw = head_crop.shape[:2]
                                max_dim = max(_hh, _hw)
                                if max_dim < 180:
                                    scale = max(2.0, 256.0 / max(1, max_dim))
                                    head_crop_scaled = cv2.resize(head_crop, (int(_hw * scale), int(_hh * scale)), interpolation=cv2.INTER_LINEAR)
                                elif max_dim > 320:
                                    scale = 288.0 / float(max_dim)
                                    head_crop_scaled = cv2.resize(head_crop, (int(_hw * scale), int(_hh * scale)), interpolation=cv2.INTER_AREA)
                                else:
                                    head_crop_scaled = head_crop
                                    scale = 1.0

                                h_res = self.ppe_model.predict(head_crop_scaled, conf=0.15, imgsz=256, verbose=False, device=self.device)
                                mask_boxes = []
                                no_mask_boxes = []
                                if h_res and len(h_res[0].boxes) > 0:
                                    for hbox in h_res[0].boxes:
                                        hcls = int(hbox.cls[0].item())
                                        hconf = float(hbox.conf[0].item())
                                        cx1, cy1, cx2, cy2 = [int(v / scale) for v in hbox.xyxy[0].tolist()]
                                        bot_rel = cy2 / float(_hh)
                                        cy_rel = (cy1 + cy2) / (2.0 * _hh)
                                        
                                        # Strict Anatomical Guard:
                                        # A face mask must cover the lower face (nose/mouth).
                                        # It CANNOT sit entirely on hair or upper forehead (bot_rel < 0.48 or cy_rel < 0.35).
                                        # Requires hconf >= 0.38 to eliminate false positives on dark wavy hair.
                                        if hcls == 3: # mask
                                            if bot_rel >= 0.45 and cy_rel >= 0.32 and hconf >= 0.35:
                                                gbox = [
                                                    max(0, min(w, hx1 + cx1)),
                                                    max(0, min(h, hy1 + cy1)),
                                                    max(0, min(w, hx1 + cx2)),
                                                    max(0, min(h, hy1 + cy2))
                                                ]
                                                mask_boxes.append((gbox, hconf))
                                        elif hcls == 7: # no_mask
                                            if bot_rel >= 0.40 and hconf >= 0.20:
                                                # Narrow no_mask box specifically over mouth/lower face
                                                nm_y1 = max(hy1 + int(_hh * 0.42), hy1 + cy1)
                                                nm_y2 = min(hy1 + int(_hh * 0.90), hy1 + cy2)
                                                nm_x1 = max(hx1 + int(_hw * 0.20), hx1 + cx1)
                                                nm_x2 = min(hx1 + int(_hw * 0.80), hx1 + cx2)
                                                gbox_nm = [max(0, min(w, nm_x1)), max(0, min(h, nm_y1)), max(0, min(w, nm_x2)), max(0, min(h, nm_y2))]
                                                no_mask_boxes.append((gbox_nm, hconf))

                                # ── Industrial 3M Cartridge Respirator Detection ──
                                # Detects 3M P100 pink/magenta cartridges on lower face
                                has_cart, cart_box, cart_conf = detect_industrial_cartridge_in_lower_face(head_crop)
                                if has_cart and cart_box is not None:
                                    # Span respirator box centered across mouth and nose
                                    mask_w = max(int(_hw * 0.52), (cart_box[2] - cart_box[0]) * 2)
                                    mask_h = max(int(_hh * 0.42), (cart_box[3] - cart_box[1]) + int(_hh * 0.15))
                                    c_cx = (cart_box[0] + cart_box[2]) // 2
                                    c_cy = (cart_box[1] + cart_box[3]) // 2
                                    m_cx = (_hw // 2 + c_cx) // 2
                                    m_cy = max(int(_hh * 0.60), c_cy)
                                    mx1 = max(0, min(_hw, m_cx - mask_w // 2))
                                    mx2 = max(0, min(_hw, m_cx + mask_w // 2))
                                    my1 = max(0, min(_hh, m_cy - mask_h // 2))
                                    my2 = max(0, min(_hh, m_cy + mask_h // 2))
                                    g_cart_box = [max(0, min(w, hx1 + mx1)), max(0, min(h, hy1 + my1)), max(0, min(w, hx1 + mx2)), max(0, min(h, hy1 + my2))]
                                    mask_boxes.append((g_cart_box, cart_conf))
                                    # Pink cartridge confirms respirator is worn; suppress weak/spurious no_mask detections
                                    no_mask_boxes = [b for b in no_mask_boxes if b[1] > 0.85]

                                now_t = time.time()
                                if tid not in self.worker_gear_memory:
                                    self.worker_gear_memory[tid] = {
                                        'last_mask_time': 0,
                                        'last_mask_box': None,
                                        'last_glove_time': 0,
                                        'last_glove_box': None
                                    }
                                mem = self.worker_gear_memory[tid]

                                if mask_boxes:
                                    best_m = max(mask_boxes, key=lambda x: x[1])
                                    best_nm = max(no_mask_boxes, key=lambda x: x[1]) if no_mask_boxes else None
                                    if best_nm is None or best_m[1] >= best_nm[1]:
                                        has_mask = True
                                        mask_box_global = best_m[0]
                                        mem['last_mask_time'] = now_t
                                        mem['last_mask_box'] = best_m[0]
                                    else:
                                        has_mask = False
                                        mask_box_global = best_nm[0]
                                        mem['last_mask_time'] = 0
                                elif no_mask_boxes:
                                    best_nm = max(no_mask_boxes, key=lambda x: x[1])
                                    if best_nm[1] >= 0.35:
                                        has_mask = False
                                        mask_box_global = best_nm[0]
                                        mem['last_mask_time'] = 0
                                    elif (now_t - mem['last_mask_time']) < 12.0 and mem['last_mask_box'] is not None:
                                        has_mask = True
                                        mask_box_global = mem['last_mask_box']
                                    else:
                                        has_mask = False
                                        mask_box_global = best_nm[0]
                                else:
                                    if (now_t - mem['last_mask_time']) < 12.0 and mem['last_mask_box'] is not None:
                                        has_mask = True
                                        mask_box_global = mem['last_mask_box']
                                    else:
                                        has_mask = False
    
                            # ── 2. Smart Multi-Zone Hand / Glove Inspection ──
                            # Focuses specifically on wrists, hands, and working palms (excludes shoulders/biceps)
                            arm_zones = [
                                # Left hand / wrist zone (lower arm down to hand)
                                (max(0, px1 - 20),              max(0, py1 + int(ph * 0.44)),
                                 min(w, px1 + int(pw * 0.42)),  min(h, py1 + int(ph * 0.90))),
                                # Right hand / wrist zone
                                (max(0, px1 + int(pw * 0.58)),  max(0, py1 + int(ph * 0.44)),
                                 min(w, px2 + 20),              min(h, py1 + int(ph * 0.90))),
                                # Working hands in front of torso / table / machinery
                                (max(0, px1 + int(pw * 0.15)),  max(0, py1 + int(ph * 0.40)),
                                 min(w, px1 + int(pw * 0.85)),  min(h, py1 + int(ph * 0.90))),
                                # Forward reaching hands
                                (max(0, px1 - 15),              max(0, py1 + int(ph * 0.40)),
                                 min(w, px2 + 15),              min(h, py1 + int(ph * 0.90))),
                            ]
                            has_gloves = False
                            has_bare_skin = False
                            glove_box_global = None
                            best_glove_conf = 0.0
                            best_noglove_conf = 0.0
    
                            for (zx1, zy1, zx2, zy2) in arm_zones:
                                z_crop = frame[zy1:zy2, zx1:zx2]
                                if z_crop.size == 0: continue
                                _zh, _zw = z_crop.shape[:2]
                                max_z = max(_zh, _zw)
                                if max_z < 90:
                                    z_scale = max(2.0, 192.0 / max(1, max_z))
                                    z_norm = cv2.resize(z_crop, (int(_zw * z_scale), int(_zh * z_scale)), interpolation=cv2.INTER_LINEAR)
                                elif max_z > 280:
                                    z_scale = 192.0 / float(max_z)
                                    z_norm = cv2.resize(z_crop, (int(_zw * z_scale), int(_zh * z_scale)), interpolation=cv2.INTER_AREA)
                                else:
                                    z_norm = z_crop
                                    z_scale = 1.0

                                z_res = self.ppe_model.predict(z_norm, conf=0.10, imgsz=192, verbose=False, device=self.device)
                                if z_res and len(z_res[0].boxes) > 0:
                                    for zb in z_res[0].boxes:
                                        zcls = int(zb.cls[0].item())
                                        zconf = float(zb.conf[0].item())
                                        acx1, acy1, acx2, acy2 = [int(v / z_scale) for v in zb.xyxy[0].tolist()]
                                        box_mid_y = zy1 + (acy1 + acy2) // 2
                                        # Only accept detections in the lower hand/palm zone
                                        if box_mid_y >= py1 + int(ph * 0.38):
                                            g_mapped = [
                                                max(0, min(w, zx1 + acx1)),
                                                max(0, min(h, zy1 + acy1)),
                                                max(0, min(w, zx1 + acx2)),
                                                max(0, min(h, zy1 + acy2))
                                            ]
                                            if zcls == 0 and zconf >= 0.20: # glove
                                                if zconf > best_glove_conf:
                                                    best_glove_conf = zconf
                                                    glove_box_global = g_mapped
                                                    has_gloves = True
                                            elif zcls == 4 and zconf >= 0.20: # no_glove
                                                if zconf > best_noglove_conf:
                                                    best_noglove_conf = zconf
                                                    glove_box_global = g_mapped
                                                    has_bare_skin = True

                                # Bare hand skin check: locates exposed palm
                                skin_detected, s_ratio, palm_box = detect_bare_hands_in_zone(z_crop)
                                if skin_detected and 0.08 < s_ratio < 0.55:
                                    has_bare_skin = True
                                    if palm_box is not None and (best_noglove_conf < 0.60):
                                        glove_box_global = [
                                            max(0, min(w, zx1 + palm_box[0])),
                                            max(0, min(h, zy1 + palm_box[1])),
                                            max(0, min(w, zx1 + palm_box[2])),
                                            max(0, min(h, zy1 + palm_box[3]))
                                        ]
                                        best_noglove_conf = 0.60
    
                            now_t = time.time()
                            if tid not in self.worker_gear_memory:
                                self.worker_gear_memory[tid] = {
                                    'last_mask_time': 0,
                                    'last_mask_box': None,
                                    'last_glove_time': 0,
                                    'last_glove_box': None
                                }
                            mem = self.worker_gear_memory[tid]

                            if has_gloves:
                                glove_status = "present"
                                mem['last_glove_time'] = now_t
                                mem['last_glove_box'] = glove_box_global
                            elif has_bare_skin:
                                glove_status = "absent"
                                mem['last_glove_time'] = 0
                            else:
                                if (now_t - mem['last_glove_time']) < 8.0 and mem['last_glove_box'] is not None:
                                    glove_status = "present"
                                    glove_box_global = mem['last_glove_box']
                                else:
                                    glove_status = "present" # Default factory standard
    
                            mask_status = "present" if has_mask else "absent"
                            is_compliant = (mask_status == "present") and (glove_status == "present")
    
                            if is_compliant:
                                compliant_count += 1
                            else:
                                violating_count += 1
    
                            current_cache[tid] = {
                                "box": [px1, py1, px2, py2],
                                "mask_status": mask_status,
                                "glove_status": glove_status,
                                "mask_box": mask_box_global,
                                "glove_box": glove_box_global,
                                "is_compliant": is_compliant
                            }
                        worker_cache = current_cache
    
                # Draw annotations
                for tid, info in worker_cache.items():
                    px1, py1, px2, py2 = info["box"]
                    is_comp = info["is_compliant"]
                    bcolor = (0, 200, 0) if is_comp else (0, 0, 240)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), bcolor, 2)
    
                    # Gear boxes
                    if info["mask_box"]:
                        mx1, my1, mx2, my2 = info["mask_box"]
                        if info["mask_status"] == "present":
                            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 200, 0), 2)
                            draw_label(frame, "MASK", (mx1, my1 - 2), (255, 140, 0))
                        else:
                            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 240), 2)
                            draw_label(frame, "NO MASK", (mx1, my1 - 2), (0, 0, 200))
                    if info["glove_box"]:
                        gx1, gy1, gx2, gy2 = info["glove_box"]
                        if info["glove_status"] == "present":
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 220, 255), 2)
                            draw_label(frame, "GLOVES", (gx1, gy1 - 2), (0, 180, 220))
                        else:
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 240), 2)
                            draw_label(frame, "NO GLOVES", (gx1, gy1 - 2), (0, 0, 200))
    
                    # Header badge
                    missing = []
                    if info["mask_status"] == "absent": missing.append("Mask")
                    if info["glove_status"] == "absent": missing.append("Gloves")
    
                    if is_comp:
                        lbl = f"Worker #{tid} - COMPLIANT"
                        draw_label(frame, lbl, (px1, max(12, py1 - 8)), (0, 150, 0))
                    else:
                        lbl = f"Worker #{tid} - NON-COMPLIANT"
                        draw_label(frame, lbl, (px1, max(12, py1 - 8)), (0, 0, 200))
                        miss_lbl = f"MISSING: {', '.join(missing)}"
                        draw_label(frame, miss_lbl, (px1, max(12, py1 - 24)), (0, 0, 200))
                        # Log event
                        now_str = datetime.now().strftime("%H:%M:%S")
                        event_key = f"{tid}-{'-'.join(missing)}"
                        if not any(e["key"] == event_key and time.time() - e["t"] < 5 for e in stream_state.recent_events):
                            stream_state.recent_events.appendleft({
                                "key": event_key,
                                "t": time.time(),
                                "time": now_str,
                                "worker_id": tid,
                                "missing": missing,
                                "camera": cam_name
                            })
    
                # Calculate FPS
                t1 = time.time()
                fps_tracker.append(1.0 / max(0.001, (t1 - t0)))
                cur_fps = round(float(np.mean(fps_tracker)), 1)
                stream_state.fps = cur_fps
                stream_state.active_workers = len(worker_cache)
                stream_state.compliant_workers = sum(1 for v in worker_cache.values() if v["is_compliant"])
                stream_state.violating_workers = stream_state.active_workers - stream_state.compliant_workers
                total_active = max(1, stream_state.active_workers)
                stream_state.compliance_rate = int((stream_state.compliant_workers / total_active) * 100)
    
                # Draw stylish on-screen HUD (subtle top status bar)
                hud_text = f"{cam_name} | {cur_fps} FPS | {self.dev_name} | {stream_state.compliance_rate}% Compliant"
                draw_label(frame, hud_text, (10, 25), (40, 40, 40), scale=0.55, thickness=1)
    
                # Encode JPEG (downscale for smooth browser bandwidth if over 1280px)
                display_frame = cv2.resize(frame, (1280, int(h * (1280 / w)))) if w > 1280 else frame
                _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                stream_state.current_frame_jpeg = frame_bytes
                stream_state.last_frame_time = time.time()
    
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            if cap is not None:
                cap.release()

# Global pipeline instance
pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = VideoPipeline()
    return pipeline_instance
