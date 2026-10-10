import os
import sys
import time
import argparse
import csv
import re
from datetime import datetime
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

# Enable ANSI escape sequences on Windows
os.system('')

# Ensure immediate unbuffered terminal prints
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

def load_env_file():
    for env_path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
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

CURRENT_CAMERA_NAME = ""

# ANSI Colors for Terminal
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
GRAY = "\033[90m"
WHITE = "\033[97m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
LOG_CSV_PATH = os.path.join(BACKEND_DIR, "violations_log.csv")

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
    'mask': (255, 255, 0),        # Cyan for Present Mask / Respirator
    'respirator': (255, 255, 0),  # Cyan
    'glove': (0, 255, 255),       # Bright Yellow for Gloves
    'helmet': (255, 191, 0),      # Blue for Helmet
    'shoes': (255, 0, 255),       # Magenta for Safety Shoes
    'goggles': (255, 255, 0),     # Cyan
    'no_mask': (0, 0, 255),       # Red for Bare Face
    'no_glove': (0, 0, 255),      # Red for Bare Hand
    'no_helmet': (0, 0, 255),     # Red
    'no_shoes': (0, 0, 255),      # Red
}

WORKER_HISTORY = {}
WORKER_GEAR_MEMORY = {}

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

worker_registry = PersistentWorkerRegistry()

def denoise_frame(frame, clahe_clip=1.5):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

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
    """
    SKIN-EXPOSURE DETECTOR for hand zones:
    Detects exposed skin (bare hand / wrist / palm) within a crop.
    Returns: (has_bare_skin, ratio_of_skin_pixels, palm_box)
    """
    if zone_crop is None or zone_crop.size == 0:
        return False, 0.0, None

    zz = cv2.GaussianBlur(zone_crop, (3, 3), 0)
    hsv = cv2.cvtColor(zz, cv2.COLOR_BGR2HSV)

    # Broad skin-tone range (covers light to medium-dark complexions)
    skin_lo  = cv2.inRange(hsv, np.array([  0, 18,  70]), np.array([ 22, 170, 255]))
    skin_hi  = cv2.inRange(hsv, np.array([165, 18,  70]), np.array([179, 170, 255]))
    skin = cv2.bitwise_or(skin_lo, skin_hi)

    # Clean noise
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


def overlaps(box1, box2, threshold=0.10):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    b1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    return inter_area / float(b1_area) if b1_area > 0 else 0.0

def smooth_compliance(track_id, current_missing):
    if track_id not in WORKER_HISTORY:
        WORKER_HISTORY[track_id] = deque(maxlen=4)
    WORKER_HISTORY[track_id].append(set(current_missing))
    history_len = len(WORKER_HISTORY[track_id])
    smoothed = []
    for item in ['Helmet', 'Mask', 'Gloves', 'Shoes']:
        count = sum(1 for missing_set in WORKER_HISTORY[track_id] if item in missing_set)
        if count >= 2 or (history_len < 2 and count == history_len):
            smoothed.append(item)
    return smoothed

def log_violation(timestamp_str, track_id, missing_items):
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_CSV_PATH)
    try:
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'video_name', 'frame_number', 'track_id', 'violation_items', 'snapshot_path'])
            writer.writerow([timestamp_str, CURRENT_CAMERA_NAME or "Live CCTV", 0, track_id, ", ".join(missing_items), ""])
    except Exception:
        pass

def is_ip_address(val):
    pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?$'
    return bool(re.match(pattern, str(val).strip()))

def parse_channel_number(ch):
    if ch is None:
        return 29
    # Strip any 'D', 'd', 'CH', 'ch', 'c' prefix (e.g. 'D29' -> 29)
    cleaned = re.sub(r'^[DdCcHh]+', '', str(ch).strip())
    try:
        return int(cleaned)
    except ValueError:
        return 29

def parse_cctv_composite_string(source_str):
    """
    Parses composite strings such as:
    http://192.168.127.5/doc/index.html#/preview,admin,dmin@123,[D29] POWDER COATING OFFICE AREA
    or 192.168.127.5,admin,dmin@123,29
    """
    s = str(source_str).strip().strip('"').strip("'")
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        ip_match = re.search(r'([0-9]{1,3}(?:\.[0-9]{1,3}){3}(?::[0-9]{1,5})?)', parts[0])
        ip = ip_match.group(1) if ip_match else parts[0]
        user = parts[1] if len(parts) > 1 and parts[1] else "admin"
        pwd = parts[2] if len(parts) > 2 else ""
        ch = 29
        area = ""
        if len(parts) > 3:
            raw_ch = parts[3]
            ch_match = re.search(r'\[?[DdCcHh]?([0-9]+)\]?', raw_ch)
            if ch_match:
                ch = int(ch_match.group(1))
            area = re.sub(r'\[.*?\]', '', raw_ch).strip()
            if not area:
                area = f"Digital Channel {ch}"
        return ip, user, pwd, ch, area
    return None

def build_cctv_candidates(ip, user="admin", password="", channel=29):
    auth = f"{user}:{password}@" if (user or password) else ""
    clean_ip = ip.split(':')[0]
    port = ip.split(':')[1] if ':' in ip else '554'
    host = f"{clean_ip}:{port}"
    ch = parse_channel_number(channel)
    return [
        (f'Hikvision Sub-Stream (D{ch} / Ch {ch}02)', f"rtsp://{auth}{host}/Streaming/Channels/{ch}02"),
        (f'Dahua / CP Plus Sub-Stream (D{ch})', f"rtsp://{auth}{host}/cam/realmonitor?channel={ch}&subtype=1"),
        (f'Uniview Sub-Stream (D{ch})', f"rtsp://{auth}{host}/unicast/c{ch}/s1/live"),
        (f'Generic ONVIF (D{ch})', f"rtsp://{auth}{host}/live/ch{ch}"),
        (f'Hikvision Main-Stream (D{ch} / Ch {ch}01)', f"rtsp://{auth}{host}/Streaming/Channels/{ch}01"),
        (f'Dahua / CP Plus Main-Stream (D{ch})', f"rtsp://{auth}{host}/cam/realmonitor?channel={ch}&subtype=0"),
        (f'HTTP MJPEG Stream (:8080)', f"http://{auth}{clean_ip}:8080/video"),
    ]

def resolve_cctv_source(source, user="admin", password="", brand="auto", channel=29, area=None):
    global CURRENT_CAMERA_NAME
    source_str = str(source).strip().strip('"').strip("'")

    parsed = parse_cctv_composite_string(source_str)
    if parsed:
        p_ip, p_user, p_pwd, p_ch, p_area = parsed
        source_str = p_ip
        if p_user: user = p_user
        if p_pwd is not None: password = p_pwd
        if p_ch: channel = p_ch
        if p_area: area = p_area

    ch_num = parse_channel_number(channel)
    if area:
        CURRENT_CAMERA_NAME = f"[D{ch_num}] {area}"
    else:
        CURRENT_CAMERA_NAME = f"Camera [D{ch_num}]"

    if is_ip_address(source_str) or source_str.startswith(("http://", "https://")):
        ip_m = re.search(r'([0-9]{1,3}(?:\.[0-9]{1,3}){3}(?::[0-9]{1,5})?)', source_str)
        clean_ip = ip_m.group(1) if ip_m else source_str
        
        print(f"\n{BOLD}{CYAN}[*] CCTV Connection: {clean_ip} | Channel: [D{ch_num}] {f'({area})' if area else ''}{RESET}")
        print(f"[*] Auto-resolving CCTV RTSP stream for Camera D{ch_num}...")
        candidates = build_cctv_candidates(clean_ip, user=user, password=password, channel=ch_num)
        
        brand_l = brand.lower().strip() if brand else "auto"
        if "hik" in brand_l:
            print(f"{GREEN}[✓] Using Hikvision Sub-Stream: rtsp://***:***@{clean_ip}:554/Streaming/Channels/{ch_num}02{RESET}\n")
            return candidates[0][1]
        elif "dahua" in brand_l or "cp" in brand_l:
            return candidates[1][1]
        elif "uni" in brand_l:
            return candidates[2][1]
        
        # Fast socket check before probing
        port = int(clean_ip.split(':')[1]) if ':' in clean_ip else 554
        pure_ip = clean_ip.split(':')[0]
        try:
            import socket
            s = socket.create_connection((pure_ip, port), timeout=0.8)
            s.close()
            port_open = True
        except Exception:
            port_open = False

        if not port_open:
            print(f"\n{RED}[ERROR] RTSP Port {port} on IP {pure_ip} is not responding.{RESET}")
            print(f"Please check:")
            print(f"  1. Is your PC connected to the CCTV / Factory Wi-Fi network?")
            print(f"  2. Verify you can ping {pure_ip} from your terminal.")
            print(f"  3. Is the RTSP service (Port 554) active on the NVR?\n")
            sys.exit(1)

        # Test candidate streams quickly
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"
        for label, url in candidates[:4]:
            masked_url = url
            if "@" in url:
                pre, post = url.split("@", 1)
                masked_url = f"{pre.split('//')[0]}//***:***@{post}"
            print(f"  * Probing {label}: {masked_url}")
            test_cap = cv2.VideoCapture(url)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"{GREEN}[✓] Successfully connected to {label}!{RESET}\n")
                    return url
            test_cap.release()
            
        print(f"{YELLOW}[!] Auto-probe fallback: Using standard Hikvision RTSP Sub-Stream:{RESET}")
        print(f"    {candidates[0][1]}\n")
        return candidates[0][1]

    return source

def select_video_interactive():
    candidates = [
        r"E:\New folder (2)\CCTV 3.mp4",
        r"E:\New folder (2)\CCTV 1.mp4",
        r"E:\New folder (2)\CCTV 2.mp4",
    ]
    existing = [p for p in candidates if os.path.exists(p)]
    
    print(f"\n{BOLD}{CYAN}=== Select Video Source ==={RESET}")
    print(f"  {BOLD}[23]{RESET} {GREEN}{BOLD}[D23] TOOL ROOM{RESET} (Live 192.168.127.5 Ch 2302)")
    print(f"  {BOLD}[3]{RESET}  {GREEN}{BOLD}[D3]  PC SPRAY BOOTH WEST WALL{RESET} (Live 192.168.127.5 Ch 302)")
    print(f"  {BOLD}[29]{RESET} {GREEN}{BOLD}[D29] POWDER COATING OFFICE AREA{RESET} (Live 192.168.127.5 Ch 2902)")
    print(f"  {BOLD}[0]{RESET}  {CYAN}Live Webcam #0 (USB / Built-in Camera){RESET}")
    print(f"  {BOLD}[R]{RESET}  {MAGENTA}Live External CCTV Server / IP Camera (Enter IP or RTSP){RESET}")
    for idx, path in enumerate(existing):
        sz = os.path.getsize(path) / (1024 * 1024)
        tag = f"{GREEN}[FINE-TUNED]{RESET}" if "CCTV 3" in path else ""
        print(f"  {BOLD}[{idx + 1}]{RESET} {path} ({sz:.1f} MB) {tag}")
    print(f"  {BOLD}[{len(existing) + 1}]{RESET} Enter custom video file path")
    
    choice = input(f"\nEnter choice [23, 3, 29, 0, R, or 1-{len(existing)+1}] (default: 23): ").strip() or "23"
    
    try:
        if choice.upper() in ["23", "D23"]:
            return resolve_cctv_source(
                "192.168.127.5",
                user="admin",
                password="dmin@123",
                brand="hikvision",
                channel=23,
                area="TOOL ROOM"
            )
        elif choice.upper() in ["3", "D3"]:
            return resolve_cctv_source(
                "192.168.127.5",
                user="admin",
                password="dmin@123",
                brand="hikvision",
                channel=3,
                area="PC SPRAY BOOTH WEST WALL"
            )
        elif choice.upper() in ["D", "D29", "29"]:
            return resolve_cctv_source(
                "192.168.127.5",
                user="admin",
                password="dmin@123",
                brand="hikvision",
                channel=29,
                area="POWDER COATING OFFICE AREA"
            )
        elif choice.upper() == "R":
            print(f"\n{BOLD}{CYAN}=== Connect to CCTV NVR / IP Camera ==={RESET}")
            ip_or_url = input("Enter NVR / Camera IP Address (e.g., 192.168.127.5) OR Full string: ").strip().strip('"').strip("'")
            if parse_cctv_composite_string(ip_or_url):
                return resolve_cctv_source(ip_or_url)
            elif is_ip_address(ip_or_url):
                ch_in = input("Enter Digital Channel / Camera Number [e.g. 29 for D29] (default: 29): ").strip() or "29"
                user = input("Enter Username (default: admin): ").strip() or "admin"
                pwd = input("Enter Password (default: dmin@123): ").strip() or "dmin@123"
                brand = input("Brand [1=Hikvision, 2=Dahua/CP Plus, 3=Uniview, 4=Auto-Detect] (default: 1): ").strip() or "1"
                brand_map = {"1": "hikvision", "2": "dahua", "3": "uniview", "4": "auto"}
                return resolve_cctv_source(ip_or_url, user=user, password=pwd, brand=brand_map.get(brand, "hikvision"), channel=ch_in)
            return ip_or_url
        elif choice == "0":
            return 0
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(existing):
            return existing[choice_idx]
        else:
            custom_path = input("Enter full path to video file or stream URL / IP: ").strip().strip('"').strip("'")
            if parse_cctv_composite_string(custom_path) or is_ip_address(custom_path):
                return resolve_cctv_source(custom_path)
            return custom_path
    except Exception:
        return 0

def draw_label(img, text, pt, bg_color, text_color=(255, 255, 255), scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pt
    x = max(5, x)
    y = max(th + 10, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

def run_cli_detector(video_source, show_gui=True, strict_mode=False, conf_threshold=0.18, tracker="bytetrack.yaml", stride=None, camera_name=None):
    global CURRENT_CAMERA_NAME
    active_name = camera_name or CURRENT_CAMERA_NAME
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{WHITE}   ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - INDUSTRY ENGINE{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    if active_name:
        print(f"[*] Monitored Camera Location: {YELLOW}{BOLD}{active_name}{RESET}")

    # Load Model Weights
    weights_path = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BACKEND_DIR, "yolov8n-ppe.pt")
    
    print(f"[*] Loading Industrial PPE Model: {YELLOW}{os.path.basename(weights_path)}{RESET}")
    ppe_model = YOLO(weights_path)
    
    tracker_factory = os.path.join(BASE_DIR, "bytetrack_factory.yaml")
    if os.path.exists(tracker_factory):
        tracker = tracker_factory

    print(f"[*] Loading Person Tracker: {YELLOW}yolo11n.pt{RESET} + {GREEN}ByteTrack ({os.path.basename(tracker)}){RESET}")
    person_model = YOLO("yolo11n.pt")

    import torch
    infer_device = 0 if torch.cuda.is_available() else 'cpu'
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Inference Hardware: {GREEN}{dev_name}{RESET}")

    video_str = str(video_source).strip()
    is_rtsp = video_str.startswith("rtsp://")
    is_http_stream = video_str.startswith(("http://", "https://"))
    is_webcam = isinstance(video_source, int) or video_str.isdigit()
    is_live = is_webcam or is_rtsp or is_http_stream

    if is_rtsp:
        # Enforce TCP transport for RTSP to prevent packet drop and frame corruption in factory networks
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        print(f"[*] RTSP Transport: {GREEN}Enforcing TCP mode (low packet loss){RESET}")

    # Open Video Source
    cap = cv2.VideoCapture(video_source if not is_webcam else int(video_source))
    if not cap.isOpened():
        print(f"{RED}[ERROR] Failed to open video source: {video_source}{RESET}")
        return

    if is_live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent buffer bloat / stream latency lag
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
    
    if is_rtsp or is_http_stream:
        src_label = f"Live CCTV [{active_name}]" if active_name else (f"Live CCTV Stream ({video_str[:35]}...)" if len(video_str) > 35 else f"Live CCTV Stream ({video_str})")
    elif is_webcam:
        src_label = f"Webcam #{video_source}"
    else:
        src_label = os.path.basename(video_str)

    print(f"[*] Source: {CYAN}{src_label}{RESET} | Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"[*] Multi-Object Tracker: {GREEN}ByteTrack ({tracker}){RESET}")
    print(f"[*] Respirator Support: {GREEN}3M 2091/2097 Pink + 3M 6001/6003 Yellow + N95/Surgical Active{RESET}")
    print(f"[*] Inspection Mode: {YELLOW}{'Strict (All 4 Items Required)' if strict_mode else 'Standard (Mask/Respirator + Gloves)'}{RESET}")
    print(f"[*] GUI Window: {GREEN if show_gui else GRAY}{'Active (Press Q to quit, P to pause)' if show_gui else 'Headless'}{RESET}")
    print(f"{CYAN}------------------------------------------------------------------------{RESET}\n")

    if stride is None:
        # Live Webcam: stride 1 (process every frame) for true maximum real-time potential.
        # Live RTSP: stride 2 (process every other frame) for stream buffer stability.
        # Video file: stride 3 for fast review speed.
        stride = 1 if is_webcam else (2 if is_live else 3)

    # Warm up models on GPU/CPU to eliminate initial inference lag
    dummy = np.zeros((576, 1024, 3), dtype=np.uint8)
    person_model.track(dummy, persist=True, classes=[0], imgsz=512, tracker=tracker, verbose=False, device=infer_device)
    ppe_model.predict(dummy, conf=0.14, imgsz=512, verbose=False, device=infer_device)

    frame_count = 0
    total_workers_seen = set()
    total_violations_logged = 0
    last_log_time = {}
    paused = False
    start_time = time.time()
    last_terminal_print = 0
    
    fps_history = deque(maxlen=20)
    prev_time = time.time()
    WORKER_GEAR_CACHE = {}
    last_global_detections = []

    WIN_TITLE = f"Roots Industrial PPE Detector - {active_name} (Press Q to Quit, P to Pause)" if active_name else "Roots Industrial PPE Detector (Press Q to Quit, P to Pause)"
    if show_gui:
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, 1280, 720)

    try:
        while True:
            if not paused:
                ret, raw_frame = cap.read()
                if not ret:
                    if not is_live and frame_count >= total_frames - 5:
                        print(f"\n{GREEN}[*] Reached end of video file.{RESET}")
                        break
                    elif is_live:
                        # Auto-reconnection for live CCTV/RTSP streams on network drops
                        print(f"\n{YELLOW}[!] Stream frame dropped or network interrupted. Attempting auto-reconnect...{RESET}")
                        time.sleep(1.0)
                        cap.release()
                        cap = cv2.VideoCapture(video_source if not is_webcam else int(video_source))
                        if is_live:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        continue
                    time.sleep(0.01)
                    continue

                if not is_live and stride > 1:
                    for _ in range(stride - 1):
                        cap.grab()

                frame_count += stride
                curr_t = time.time()
                instant_fps = 1.0 / max(0.001, (curr_t - prev_time))
                prev_time = curr_t
                fps_history.append(instant_fps)
                smooth_fps = sum(fps_history) / len(fps_history)

                # Upscale small/low-res CCTV frames to at least 1280px wide for display
                # and accurate tracking of small/distant workers.
                orig_h, orig_w = raw_frame.shape[:2]
                target_w = max(orig_w, 1280)  # ALWAYS ensure at least 1280px for detection
                scale_w = target_w / float(orig_w)
                target_h = int(orig_h * scale_w)
                frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                # Apply CLAHE to equalize industrial contrast for distant workers in dark areas
                frame_enhanced = denoise_frame(frame, clahe_clip=1.8)

                h, w = frame.shape[:2]

                # 1. Multi-Person Tracking with ByteTrack
                # Dynamically use 1536px for 4K / 2K / 1080p CCTV footage to keep distant workers clearly detected
                imgsz_track = 1536 if (w >= 1920 or h >= 1080) else 1280
                track_results = person_model.track(
                    frame_enhanced,
                    persist=True,
                    classes=[0],
                    conf=0.15,
                    imgsz=imgsz_track,
                    tracker=tracker,
                    verbose=False,
                    device=infer_device
                )

                # Skip full-frame global PPE inference — per-worker crops handle
                # this with better accuracy and much less CPU cost.
                global_detections = []

                current_workers_status = []
                frame_violations = 0
                frame_compliant = 0

                if track_results and track_results[0].boxes and len(track_results[0].boxes) > 0:
                    raw_boxes = track_results[0].boxes.xyxy.cpu().numpy()
                    raw_confs = track_results[0].boxes.conf.cpu().numpy()
                    if track_results[0].boxes.id is not None:
                        raw_ids = track_results[0].boxes.id.int().cpu().numpy()
                    else:
                        raw_ids = np.arange(1, len(raw_boxes) + 1, dtype=int)

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
                        for tid in list(WORKER_GEAR_CACHE.keys()):
                            WORKER_GEAR_CACHE.pop(tid, None)
                        continue

                    boxes = raw_boxes[valid_indices]
                    track_ids = [raw_ids[i] for i in valid_indices]

                    # Persistent Worker Registry: Locks transient ByteTrack IDs (e.g. 126) to permanent Worker #1, #2
                    worker_slot_ids = worker_registry.update(boxes, track_ids, time.time(), w, h)

                    active_count = len(worker_slot_ids)
                    inspect_target_idx = (frame_count % active_count) if active_count > 0 else 0

                    for idx, (box, track_id) in enumerate(zip(boxes, worker_slot_ids)):
                        total_workers_seen.add(track_id)
                        px1, py1, px2, py2 = map(int, box)
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)

                        # Reinspect all active workers regularly without round-robin bottleneck
                        should_reinspect = (track_id not in WORKER_GEAR_CACHE) or (frame_count % 2 == 0)

                        if should_reinspect:
                            gear_states = {
                                'helmet': 'unknown',
                                'mask': 'absent',
                                'glove': 'unknown',
                                'shoes': 'unknown'
                            }
                            worker_items = []

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

                                h_res = ppe_model.predict(head_crop_scaled, conf=0.15, imgsz=256, verbose=False, device=infer_device)
                                mask_boxes = []
                                no_mask_boxes = []
                                if h_res and len(h_res) > 0:
                                    for hbox in h_res[0].boxes:
                                        hcls = int(hbox.cls[0].item())
                                        hlabel = PPE_CLASSES.get(hcls, '')
                                        hconf = float(hbox.conf[0].item())
                                        cx1, cy1, cx2, cy2 = [int(v / scale) for v in hbox.xyxy[0].tolist()]
                                        bot_rel = cy2 / float(_hh)
                                        cy_rel = (cy1 + cy2) / (2.0 * _hh)
                                        
                                        # Strict Anatomical Guard:
                                        # A face mask must cover the lower face (nose/mouth).
                                        # It CANNOT sit entirely on hair or upper forehead (bot_rel < 0.48 or cy_rel < 0.35).
                                        # Requires hconf >= 0.38 to eliminate false positives on dark wavy hair.
                                        if hlabel == 'mask':
                                            if bot_rel >= 0.45 and cy_rel >= 0.32 and hconf >= 0.35:
                                                global_item_box = [max(0, min(w, hx1 + cx1)), max(0, min(h, hy1 + cy1)), max(0, min(w, hx1 + cx2)), max(0, min(h, hy1 + cy2))]
                                                mask_boxes.append((global_item_box, "mask", hconf))
                                        elif hlabel == 'no_mask':
                                            if bot_rel >= 0.40 and hconf >= 0.20:
                                                nm_y1 = max(hy1 + int(_hh * 0.42), hy1 + cy1)
                                                nm_y2 = min(hy1 + int(_hh * 0.90), hy1 + cy2)
                                                nm_x1 = max(hx1 + int(_hw * 0.20), hx1 + cx1)
                                                nm_x2 = min(hx1 + int(_hw * 0.80), hx1 + cx2)
                                                global_item_box = [max(0, min(w, nm_x1)), max(0, min(h, nm_y1)), max(0, min(w, nm_x2)), max(0, min(h, nm_y2))]
                                                no_mask_boxes.append((global_item_box, "no_mask", hconf))

                                # ── Industrial 3M Cartridge Respirator Detection ──
                                # Detects 3M P100 pink/magenta cartridges on lower face
                                has_cart, cart_box, cart_conf = detect_industrial_cartridge_in_lower_face(head_crop)
                                if has_cart and cart_box is not None:
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
                                    mask_boxes.append((g_cart_box, "mask", cart_conf))
                                    # Pink cartridge confirms respirator is worn; suppress weak/spurious no_mask detections
                                    no_mask_boxes = [b for b in no_mask_boxes if b[2] > 0.85]

                                now_t = time.time()
                                if track_id not in WORKER_GEAR_MEMORY:
                                    WORKER_GEAR_MEMORY[track_id] = {
                                        'last_mask_time': 0,
                                        'last_mask_item': None,
                                        'last_glove_time': 0,
                                        'last_glove_item': None
                                    }
                                mem = WORKER_GEAR_MEMORY[track_id]

                                if mask_boxes:
                                    best_m = max(mask_boxes, key=lambda x: x[2])
                                    best_nm = max(no_mask_boxes, key=lambda x: x[2]) if no_mask_boxes else None
                                    if best_nm is None or best_m[2] >= best_nm[2]:
                                        gear_states['mask'] = 'present'
                                        worker_items.append(best_m)
                                        mem['last_mask_time'] = now_t
                                        mem['last_mask_item'] = best_m
                                    else:
                                        gear_states['mask'] = 'absent'
                                        worker_items.append(best_nm)
                                        mem['last_mask_time'] = 0
                                elif no_mask_boxes:
                                    best_nm = max(no_mask_boxes, key=lambda x: x[2])
                                    if best_nm[2] >= 0.35:
                                        gear_states['mask'] = 'absent'
                                        worker_items.append(best_nm)
                                        mem['last_mask_time'] = 0
                                    elif (now_t - mem['last_mask_time']) < 12.0 and mem['last_mask_item'] is not None:
                                        gear_states['mask'] = 'present'
                                        worker_items.append(mem['last_mask_item'])
                                    else:
                                        gear_states['mask'] = 'absent'
                                        worker_items.append(best_nm)
                                else:
                                    # Neither mask nor no_mask visible (worker facing away / turned towards equipment)
                                    if (now_t - mem['last_mask_time']) < 12.0 and mem['last_mask_item'] is not None:
                                        gear_states['mask'] = 'present'
                                        worker_items.append(mem['last_mask_item'])
                                    else:
                                        gear_states['mask'] = 'absent'

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

                            glove_votes = 0
                            no_glove_votes = 0
                            best_glove_box = None
                            best_glove_conf = 0.0
                            best_no_glove_box = None
                            best_no_glove_conf = 0.0

                            for (zx1, zy1, zx2, zy2) in arm_zones:
                                zone_crop = frame[zy1:zy2, zx1:zx2]
                                if zone_crop.size == 0:
                                    continue
                                _zh, _zw = zone_crop.shape[:2]
                                max_z = max(_zh, _zw)
                                if max_z < 90:
                                    z_scale = max(2.0, 192.0 / max(1, max_z))
                                    zone_norm = cv2.resize(zone_crop, (int(_zw * z_scale), int(_zh * z_scale)), interpolation=cv2.INTER_LINEAR)
                                elif max_z > 280:
                                    z_scale = 192.0 / float(max_z)
                                    zone_norm = cv2.resize(zone_crop, (int(_zw * z_scale), int(_zh * z_scale)), interpolation=cv2.INTER_AREA)
                                else:
                                    zone_norm = zone_crop
                                    z_scale = 1.0

                                a_res = ppe_model.predict(
                                    zone_norm, conf=0.10, imgsz=192, verbose=False, device=infer_device)
                                if a_res and len(a_res) > 0:
                                    for abox in a_res[0].boxes:
                                        acls   = int(abox.cls[0].item())
                                        alabel = PPE_CLASSES.get(acls, '')
                                        aconf  = float(abox.conf[0].item())
                                        acx1, acy1, acx2, acy2 = [int(v / z_scale) for v in abox.xyxy[0].tolist()]
                                        box_mid_y = zy1 + (acy1 + acy2) // 2
                                        # Only accept detections in the lower hand/palm zone
                                        if box_mid_y >= py1 + int(ph * 0.38):
                                            gb = [max(0, min(w, zx1 + acx1)), max(0, min(h, zy1 + acy1)), max(0, min(w, zx1 + acx2)), max(0, min(h, zy1 + acy2))]

                                            if alabel == 'glove' and aconf >= 0.20:
                                                glove_votes += 1
                                                if aconf > best_glove_conf:
                                                    best_glove_conf = aconf
                                                    best_glove_box = gb
                                            elif alabel == 'no_glove' and aconf >= 0.20:
                                                no_glove_votes += 1
                                                if aconf > best_no_glove_conf:
                                                    best_no_glove_conf = aconf
                                                    best_no_glove_box = gb

                                # Skin-exposure check: bare skin in lower hand zone = no glove
                                has_skin, skin_ratio, palm_box = detect_bare_hands_in_zone(zone_crop)
                                if has_skin and 0.08 < skin_ratio < 0.55:
                                    no_glove_votes += 1
                                    if palm_box is not None and best_no_glove_conf < 0.60:
                                        best_no_glove_box = [
                                            max(0, min(w, zx1 + palm_box[0])),
                                            max(0, min(h, zy1 + palm_box[1])),
                                            max(0, min(w, zx1 + palm_box[2])),
                                            max(0, min(h, zy1 + palm_box[3]))
                                        ]
                                        best_no_glove_conf = 0.60

                            now_t = time.time()
                            if track_id not in WORKER_GEAR_MEMORY:
                                WORKER_GEAR_MEMORY[track_id] = {
                                    'last_mask_time': 0,
                                    'last_mask_item': None,
                                    'last_glove_time': 0,
                                    'last_glove_item': None
                                }
                            mem = WORKER_GEAR_MEMORY[track_id]

                            if glove_votes > 0:
                                gear_states['glove'] = 'present'
                                if best_glove_box:
                                    g_item = (best_glove_box, "glove", best_glove_conf)
                                    worker_items.append(g_item)
                                    mem['last_glove_time'] = now_t
                                    mem['last_glove_item'] = g_item
                            elif no_glove_votes > 0:
                                gear_states['glove'] = 'absent'
                                if best_no_glove_box:
                                    worker_items.append((best_no_glove_box, "no_glove", best_no_glove_conf))
                                mem['last_glove_time'] = 0
                            else:
                                if (now_t - mem['last_glove_time']) < 8.0 and mem['last_glove_item'] is not None:
                                    gear_states['glove'] = 'present'
                                    worker_items.append(mem['last_glove_item'])
                                else:
                                    gear_states['glove'] = 'present'

                            WORKER_GEAR_CACHE[track_id] = (gear_states, worker_items)
                        else:
                            gear_states, worker_items = WORKER_GEAR_CACHE[track_id]

                        # Determine Missing Items
                        current_missing = []
                        if strict_mode:
                            if gear_states['helmet'] != 'present':
                                current_missing.append('Helmet')
                            if gear_states['mask'] != 'present':
                                current_missing.append('Mask')
                            if gear_states['glove'] != 'present':
                                current_missing.append('Gloves')
                            if gear_states['shoes'] == 'absent':
                                current_missing.append('Shoes')
                        else:
                            # Standard factory mode
                            if gear_states['mask'] == 'absent':
                                current_missing.append('Mask')
                            elif gear_states['mask'] != 'present':
                                # Mask is required
                                current_missing.append('Mask')
                            if gear_states['glove'] == 'absent':
                                current_missing.append('Gloves')

                        smoothed_missing = smooth_compliance(track_id, current_missing)
                        is_compliant = len(smoothed_missing) == 0

                        if is_compliant:
                            frame_compliant += 1
                        else:
                            frame_violations += 1

                        current_workers_status.append({
                            'id': track_id,
                            'compliant': is_compliant,
                            'missing': smoothed_missing,
                            'gear': gear_states,
                            'box': box,
                            'items': worker_items
                        })

                        # Throttle CSV violation logs per worker (max 1 log every 4 seconds)
                        now = time.time()
                        if not is_compliant and (now - last_log_time.get(track_id, 0)) > 4.0:
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_violation(timestamp_str, track_id, smoothed_missing)
                            last_log_time[track_id] = now
                            total_violations_logged += 1
                            alert_msg = f"{RED}{BOLD}>>> [VIOLATION ALERT]{RESET} Worker #{track_id} Non-Compliant | Missing: {', '.join(smoothed_missing)}"
                            print(alert_msg)

                # Render GUI preview window
                if show_gui:
                    for w_info in current_workers_status:
                        px1, py1, px2, py2 = map(int, w_info['box'])
                        is_compliant = w_info['compliant']
                        box_color = (0, 255, 0) if is_compliant else (0, 0, 255)
                        
                        # Worker Box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                        worker_tag = f"Worker #{w_info['id']} - {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}"
                        draw_label(frame, worker_tag, (px1, py1 - 8), box_color, (255, 255, 255), scale=0.6, thickness=2)

                        # Missing Items Banner
                        if not is_compliant:
                            miss_text = f"MISSING: {', '.join(w_info['missing'])}"
                            draw_label(frame, miss_text, (px1, py1 + 22), (0, 0, 255), (255, 255, 255), scale=0.55, thickness=2)

                        # Individual item boxes — show ONLY mask/glove related items.
                        # Respirator and mask are unified under the 'mask' label with a CYAN box.
                        # No-glove / no-mask shown in RED. All other items (helmet, shoes) hidden.
                        SHOW_LABELS = {'mask', 'respirator', 'no_mask', 'glove', 'no_glove'}
                        seen_mask_box = False  # Prevent double mask/respirator box per worker
                        for (item_box, label_name, conf) in w_info['items']:
                            if label_name not in SHOW_LABELS:
                                continue
                            # Unify respirator label as MASK for display
                            display_label = "MASK" if label_name == 'respirator' else label_name.upper().replace('_', ' ')
                            # Skip duplicate mask box if we already drew one
                            if label_name in ('mask', 'respirator'):
                                if seen_mask_box:
                                    continue
                                seen_mask_box = True
                            ix1, iy1, ix2, iy2 = map(int, item_box)
                            if label_name in ('mask', 'respirator'):
                                item_color = (0, 220, 220)  # Bright cyan for present mask
                                text_c = (0, 0, 0)
                            elif label_name == 'glove':
                                item_color = (0, 255, 180)  # Mint green for gloves
                                text_c = (0, 0, 0)
                            else:
                                item_color = (0, 0, 255)    # Red for missing
                                text_c = (255, 255, 255)
                            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), item_color, 2)
                            draw_label(frame, f"{display_label} {conf:.2f}", (ix1, iy1 - 4), item_color, text_c, scale=0.48, thickness=1)

                    # Top stats bar on video window
                    cam_tag = f"CAM: {active_name} | " if active_name else ""
                    stats_str = f"{cam_tag}FPS: {smooth_fps:.1f} | ByteTrack: Active | Workers: {len(current_workers_status)} | Logged: {total_violations_logged}"
                    draw_label(frame, stats_str, (10, 25), (40, 40, 40), (0, 255, 255), scale=0.55, thickness=1)

                    cv2.imshow(WIN_TITLE, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(f"\n{YELLOW}[*] Quit requested by user.{RESET}")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"\n{YELLOW}[*] {'PAUSED' if paused else 'RESUMED'}{RESET}")

                # Terminal Dashboard Update (every 0.5s)
                now = time.time()
                if now - last_terminal_print > 0.5:
                    elapsed = now - start_time
                    sec_video = frame_count / fps if not is_live else elapsed
                    time_code = f"{int(sec_video//3600):02d}:{int((sec_video%3600)//60):02d}:{int(sec_video%60):02d}"

                    print(f"\n{BOLD}{CYAN}--- [Frame {frame_count:05d}{f'/{total_frames}' if not is_live else ''} | Time: {time_code} | Speed: {smooth_fps:.1f} FPS] ---{RESET}")
                    
                    if current_workers_status:
                        print(f"{BOLD}{'WORKER':<12} | {'RESPIRATOR/MASK':<18} | {'GLOVES':<12} | {'STATUS':<15} | {'NOTES'}{RESET}")
                        print(f"{GRAY}{'-'*78}{RESET}")
                        for w_info in current_workers_status:
                            wid = f"Worker #{w_info['id']}"
                            g = w_info['gear']
                            
                            mask_txt = f"{GREEN}[OK]{RESET}" if g['mask'] == 'present' else f"{RED}[MISSING]{RESET}"
                            glove_txt = f"{GREEN}[OK]{RESET}" if g['glove'] == 'present' else f"{RED}[MISSING]{RESET}"
                            
                            if w_info['compliant']:
                                status_txt = f"{GREEN}{BOLD}COMPLIANT{RESET}"
                                notes_txt = f"{GREEN}All Required Gear Active{RESET}"
                            else:
                                status_txt = f"{RED}{BOLD}VIOLATION{RESET}"
                                notes_txt = f"{RED}Missing: {', '.join(w_info['missing'])}{RESET}"
                                
                            print(f"{wid:<12} | {mask_txt:<27} | {glove_txt:<21} | {status_txt:<24} | {notes_txt}")
                    else:
                        print(f"{GRAY}No workers currently active in camera field of view.{RESET}")

                    total_active = len(current_workers_status)
                    compliance_rate = (frame_compliant / total_active * 100) if total_active > 0 else 100.0
                    rate_color = GREEN if compliance_rate >= 80 else (YELLOW if compliance_rate >= 50 else RED)
                    print(f"{BOLD}Summary:{RESET} Active: {total_active} | Compliant: {GREEN}{frame_compliant}{RESET} | Violations: {RED}{frame_violations}{RESET} | Rate: {rate_color}{compliance_rate:.1f}%{RESET} | Logged: {YELLOW}{total_violations_logged}{RESET}")
                    
                    last_terminal_print = now
            else:
                if show_gui:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('p'):
                        paused = not paused
                        print(f"\n{YELLOW}[*] RESUMED{RESET}")
                    elif key == ord('q'):
                        break

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[*] Monitoring interrupted by user (Ctrl+C).{RESET}")

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()

    # Final Report
    elapsed_total = time.time() - start_time
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{WHITE}                   FINAL COMPLIANCE REPORT{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"  * Total Frames Analyzed : {frame_count}")
    print(f"  * Total Time Elapsed    : {elapsed_total:.1f} seconds ({frame_count/max(0.1, elapsed_total):.1f} avg FPS)")
    print(f"  * Distinct Workers Tracked: {len(total_workers_seen)}")
    print(f"  * Total Violations Logged : {total_violations_logged}")
    print(f"  * Violations Log Path   : {LOG_CSV_PATH}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roots Industrial PPE Compliance Detector - Industry Engine")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to video file, webcam index, or RTSP/IP stream")
    parser.add_argument("--ip", type=str, default=os.getenv("CCTV_NVR_IP", None), help="Direct CCTV NVR / IP address (default: CCTV_NVR_IP env or 192.168.127.5)")
    parser.add_argument("--channel", "-ch", type=str, default=os.getenv("DEFAULT_CAMERA_CHANNEL", "29"), help="Digital camera channel on NVR [e.g. 29 for D29] (default: 29)")
    parser.add_argument("--user", "-u", type=str, default=os.getenv("CCTV_NVR_USER", "admin"), help="CCTV camera username (default: admin)")
    parser.add_argument("--password", "-p", type=str, default=os.getenv("CCTV_NVR_PASS", "dmin@123"), help="CCTV camera password (default: dmin@123)")
    parser.add_argument("--brand", "-b", type=str, default=os.getenv("CCTV_BRAND", "hikvision"), help="CCTV brand: hikvision, dahua, uniview, axis, auto")
    parser.add_argument("--area", "-a", type=str, default="POWDER COATING OFFICE AREA", help="Monitored camera area label")
    parser.add_argument("--headless", action="store_true", help="Run in pure terminal mode without OpenCV GUI window")
    parser.add_argument("--strict", "-s", action="store_true", help="Require all 4 items including helmet and shoes")
    parser.add_argument("--conf", "-c", type=float, default=0.18, help="Worker person detection confidence (default: 0.18)")
    parser.add_argument("--tracker", "-t", type=str, default="bytetrack.yaml", help="Multi-object tracker configuration (default: bytetrack.yaml)")
    parser.add_argument("--stride", type=int, default=None, help="Frame processing stride (default: 1 for webcam/RTSP, 2 for video files)")
    args = parser.parse_args()

    if args.ip:
        video_input = resolve_cctv_source(
            args.ip,
            user=args.user,
            password=args.password,
            brand=args.brand,
            channel=args.channel,
            area=args.area
        )
    elif args.video is not None:
        video_input = args.video
        if parse_cctv_composite_string(video_input) or is_ip_address(video_input) or str(video_input).startswith(("http://", "https://")):
            video_input = resolve_cctv_source(
                video_input,
                user=args.user,
                password=args.password,
                brand=args.brand,
                channel=args.channel,
                area=args.area
            )
        elif str(video_input).isdigit():
            video_input = int(video_input)
    else:
        video_input = select_video_interactive()

    run_cli_detector(
        video_source=video_input,
        show_gui=not args.headless,
        strict_mode=args.strict,
        conf_threshold=args.conf,
        tracker=args.tracker,
        stride=args.stride,
        camera_name=args.area or CURRENT_CAMERA_NAME
    )
