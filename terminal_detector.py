import os
import sys
import time
import argparse
import csv
from datetime import datetime
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

# Enable ANSI escape sequences on Windows
os.system('')

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
    'mask': (255, 255, 0),      # Cyan
    'glove': (0, 255, 255),     # Yellow
    'helmet': (255, 191, 0),    # Blue
    'shoes': (255, 0, 255),     # Magenta
    'goggles': (255, 255, 0),   # Cyan
    'no_mask': (0, 0, 255),     # Red
    'no_glove': (0, 0, 255),    # Red
    'no_helmet': (0, 0, 255),   # Red
    'no_shoes': (0, 0, 255),    # Red
}

WORKER_HISTORY = {}

def denoise_frame(frame, clahe_clip=1.5):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def detect_pink_respirator(head_crop):
    """
    Detects industrial 3M 2091/2097 magenta-pink dual-cartridge respirators.
    Only matches pure magenta/purple-pink (H: 142-168) strictly to prevent false positives on skin/walls.
    """
    if head_crop is None or head_crop.size == 0:
        return False, None
    h, w = head_crop.shape[:2]
    # Check lower half of face crop (where respirator is worn, between nose and chin)
    lower_face = head_crop[int(h * 0.35):, :]
    if lower_face.size == 0:
        return False, None

    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
    # Strictly magenta-pink (H: 142-168, S: 80-255, V: 70-255)
    pink_mask = cv2.inRange(hsv, np.array([142, 80, 70]), np.array([168, 255, 255]))
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250: # Real filter cartridge size
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Map back to head crop coords
            return True, [x, y + int(h * 0.35), x + cw, y + int(h * 0.35) + ch]
    return False, None

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
                writer.writerow(['timestamp', 'worker_id', 'missing_items'])
            writer.writerow([timestamp_str, f"Worker #{track_id}", ", ".join(missing_items)])
    except Exception:
        pass

def select_video_interactive():
    candidates = [
        r"E:\New folder (2)\CCTV 3.mp4",
        r"E:\New folder (2)\CCTV 1.mp4",
        r"E:\New folder (2)\CCTV 2.mp4",
    ]
    existing = [p for p in candidates if os.path.exists(p)]
    
    print(f"\n{BOLD}{CYAN}=== Select Video Source ==={RESET}")
    print(f"  {BOLD}[0]{RESET} {GREEN}Live Webcam #0 (USB / Built-in Camera){RESET}")
    for idx, path in enumerate(existing):
        sz = os.path.getsize(path) / (1024 * 1024)
        tag = f"{GREEN}[FINE-TUNED]{RESET}" if "CCTV 3" in path else ""
        print(f"  {BOLD}[{idx + 1}]{RESET} {path} ({sz:.1f} MB) {tag}")
    print(f"  {BOLD}[{len(existing) + 1}]{RESET} Enter custom video file path")
    
    choice = input(f"\nEnter choice [0-{len(existing)+1}] (default: 0 for Webcam): ").strip() or "0"
    
    try:
        if choice == "0":
            return 0
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(existing):
            return existing[choice_idx]
        else:
            custom_path = input("Enter full path to video file: ").strip().strip('"').strip("'")
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

def run_cli_detector(video_source, show_gui=True, strict_mode=False, conf_threshold=0.35, is_factory=False):
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{WHITE}   ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - TERMINAL ENGINE{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")

    # Load Model Weights
    weights_path = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BACKEND_DIR, "yolov8n-ppe.pt")
    
    print(f"[*] Loading PPE Model: {YELLOW}{os.path.basename(weights_path)}{RESET}")
    ppe_model = YOLO(weights_path)
    
    print(f"[*] Loading Person Tracker: {YELLOW}yolo11n.pt{RESET}")
    person_model = YOLO("yolo11n.pt")

    # Open Video Source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"{RED}[ERROR] Failed to open video source: {video_source}{RESET}")
        return

    # Optimize webcam buffer & resolution for high FPS
    is_live = isinstance(video_source, int) or str(video_source).isdigit()
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
    
    src_label = f"Webcam #{video_source}" if is_live else os.path.basename(str(video_source))
    print(f"[*] Source: {CYAN}{src_label}{RESET} | Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"[*] Detection Mode: {YELLOW}{'Factory Dual-Respirator' if is_factory else 'General Mask & PPE (Webcam/Standard)'}{RESET}")
    print(f"[*] Inspection Mode: {YELLOW}{'Strict (Helmet + Mask + Gloves + Shoes)' if strict_mode else 'Standard (Mask + Gloves)'}{RESET}")
    print(f"[*] GUI Window: {GREEN if show_gui else GRAY}{'Active (Press Q to quit, P to pause)' if show_gui else 'Headless'}{RESET}")
    print(f"{CYAN}------------------------------------------------------------------------{RESET}\n")

    frame_count = 0
    total_workers_seen = set()
    total_violations_logged = 0
    last_log_time = {}
    paused = False
    start_time = time.time()
    last_terminal_print = 0
    
    # Cache previous detections for smooth multi-frame interpolation
    cached_workers_status = []
    fps_history = deque(maxlen=20)
    prev_time = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not is_live and frame_count >= total_frames - 5:
                        print(f"\n{GREEN}[*] Reached end of video file.{RESET}")
                        break
                    time.sleep(0.01)
                    continue

                frame_count += 1
                curr_t = time.time()
                instant_fps = 1.0 / max(0.001, (curr_t - prev_time))
                prev_time = curr_t
                fps_history.append(instant_fps)
                smooth_fps = sum(fps_history) / len(fps_history)

                # Process detection on every frame (fast single-pass mode)
                h, w = frame.shape[:2]

                # 1. Global Person Tracking at 480px for high throughput
                track_results = person_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=conf_threshold,
                    imgsz=480,
                    verbose=False,
                    device='cpu'
                )

                # 2. Global PPE Inference at 480px with robust confidence (conf=0.25)
                # This prevents beards, shadows, and orange background walls from triggering false masks!
                ppe_results = ppe_model.predict(frame, conf=0.25, imgsz=480, verbose=False, device='cpu')
                
                global_detections = []
                if ppe_results and len(ppe_results) > 0:
                    for box in ppe_results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = list(map(int, box.xyxy[0].tolist()))
                        global_detections.append((xyxy, cls_id, conf))

                current_workers_status = []
                frame_violations = 0
                frame_compliant = 0

                if track_results and track_results[0].boxes and track_results[0].boxes.id is not None:
                    boxes = track_results[0].boxes.xyxy.cpu().numpy()
                    track_ids = track_results[0].boxes.id.int().cpu().numpy()

                    for box, track_id in zip(boxes, track_ids):
                        total_workers_seen.add(track_id)
                        px1, py1, px2, py2 = map(int, box)
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)

                        gear_states = {
                            'helmet': 'absent',
                            'mask': 'absent',
                            'glove': 'absent',
                            'shoes': 'unknown'
                        }
                        worker_items = []

                        # Match PPE items that overlap with this worker
                        for (box_ppe, cls_id, conf) in global_detections:
                            if overlaps(box_ppe, box, threshold=0.10) > 0.10:
                                label = PPE_CLASSES.get(cls_id, '')
                                if not label:
                                    continue
                                worker_items.append((box_ppe, cls_id, conf))
                                
                                # Exact class mapping
                                if label == 'mask':
                                    gear_states['mask'] = 'present'
                                elif label == 'no_mask':
                                    gear_states['mask'] = 'absent'
                                elif label == 'glove':
                                    gear_states['glove'] = 'present'
                                elif label == 'no_glove':
                                    gear_states['glove'] = 'absent'
                                elif label == 'shoes':
                                    gear_states['shoes'] = 'present'
                                elif label == 'no_shoes':
                                    gear_states['shoes'] = 'absent'
                                elif label == 'helmet':
                                    gear_states['helmet'] = 'present'
                                elif label == 'no_helmet':
                                    gear_states['helmet'] = 'absent'

                        # Optional Factory Pink Dual-Cartridge Respirator Inspection
                        if is_factory and gear_states['mask'] != 'present':
                            hy1 = max(0, py1 - 5)
                            hy2 = min(h, py1 + int(ph * 0.40))
                            hx1 = max(0, px1 - 10)
                            hx2 = min(w, px2 + 10)
                            head_crop = frame[hy1:hy2, hx1:hx2]
                            has_pink, pink_rect = detect_pink_respirator(head_crop)
                            if has_pink and pink_rect:
                                gear_states['mask'] = 'present'
                                gx1 = hx1 + pink_rect[0]
                                gy1 = hy1 + pink_rect[1]
                                gx2 = hx1 + pink_rect[2]
                                gy2 = hy1 + pink_rect[3]
                                worker_items.append(([gx1, gy1, gx2, gy2], 3, 0.95))

                        # Determine Missing Items
                        current_missing = []
                        if strict_mode and gear_states['helmet'] != 'present':
                            current_missing.append('Helmet')
                        if gear_states['mask'] != 'present':
                            current_missing.append('Mask')
                        if gear_states['glove'] != 'present':
                            current_missing.append('Gloves')
                        if strict_mode and gear_states['shoes'] == 'absent':
                            current_missing.append('Shoes')

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

                cached_workers_status = current_workers_status

                # Render GUI preview window
                if show_gui:
                    for w_info in cached_workers_status:
                        px1, py1, px2, py2 = map(int, w_info['box'])
                        is_compliant = w_info['compliant']
                        box_color = (0, 255, 0) if is_compliant else (0, 0, 255) # Green or Red
                        
                        # Worker Box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                        worker_tag = f"Worker #{w_info['id']} - {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}"
                        draw_label(frame, worker_tag, (px1, py1 - 8), box_color, (255, 255, 255), scale=0.6, thickness=2)

                        # Missing Items Banner
                        if not is_compliant:
                            miss_text = f"MISSING: {', '.join(w_info['missing'])}"
                            draw_label(frame, miss_text, (px1, py1 + 22), (0, 0, 255), (255, 255, 255), scale=0.55, thickness=2)

                        # Individual item boxes
                        for (item_box, cls_id, conf) in w_info['items']:
                            label_name = PPE_CLASSES.get(cls_id, '')
                            if label_name:
                                ix1, iy1, ix2, iy2 = map(int, item_box)
                                item_color = COLOR_PALETTE.get(label_name, (255, 255, 255))
                                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), item_color, 2)
                                text_c = (0, 0, 0) if label_name == 'glove' else (255, 255, 255)
                                draw_label(frame, f"{label_name} {conf:.2f}", (ix1, iy1 - 4), item_color, text_c, scale=0.45, thickness=1)

                    # Top stats bar on video window
                    stats_str = f"FPS: {smooth_fps:.1f} | Workers: {len(cached_workers_status)} | Violations: {total_violations_logged}"
                    draw_label(frame, stats_str, (10, 25), (40, 40, 40), (0, 255, 255), scale=0.55, thickness=1)

                    cv2.imshow("Roots Industrial PPE Detector (Press Q to Quit, P to Pause)", frame)
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
                    
                    if cached_workers_status:
                        print(f"{BOLD}{'WORKER':<12} | {'RESPIRATOR/MASK':<16} | {'GLOVES':<12} | {'STATUS':<15} | {'NOTES'}{RESET}")
                        print(f"{GRAY}{'-'*75}{RESET}")
                        for w_info in cached_workers_status:
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
                                
                            print(f"{wid:<12} | {mask_txt:<25} | {glove_txt:<21} | {status_txt:<24} | {notes_txt}")
                    else:
                        print(f"{GRAY}No workers currently active in camera field of view.{RESET}")

                    total_active = len(cached_workers_status)
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
    parser = argparse.ArgumentParser(description="Roots Industrial PPE Compliance Detector - Terminal Engine")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to video file or webcam index (default: interactive prompt)")
    parser.add_argument("--headless", action="store_true", help="Run in pure terminal mode without OpenCV GUI window")
    parser.add_argument("--strict", "-s", action="store_true", help="Require all 4 items including helmet and shoes")
    parser.add_argument("--conf", "-c", type=float, default=0.35, help="Worker person detection confidence")
    parser.add_argument("--factory", action="store_true", help="Enable 3M pink dual-cartridge respirator color verification for factory CCTV footage")
    args = parser.parse_args()

    video_input = args.video
    if video_input is None:
        video_input = select_video_interactive()
    else:
        if video_input.isdigit():
            video_input = int(video_input)

    run_cli_detector(
        video_source=video_input,
        show_gui=not args.headless,
        strict_mode=args.strict,
        conf_threshold=args.conf,
        is_factory=args.factory
    )
