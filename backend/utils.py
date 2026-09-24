import cv2
import numpy as np
import os
import time
import csv
import imageio
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# PPE Model Classes mapping
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

# Color palette for distinct individual PPE item bounding boxes
COLOR_PALETTE = {
    'mask': (255, 255, 0),      # Cyan for Present Mask / Respirator
    'glove': (0, 255, 255),     # Bright Yellow for Present Glove
    'helmet': (255, 191, 0),    # Blue for Present Helmet
    'shoes': (255, 0, 255),     # Magenta for Present Shoes
    'goggles': (255, 255, 0),   # Cyan for Respirator/Goggles
    'no_mask': (0, 0, 255),     # Bright Red for Bare Face / Missing Mask
    'no_glove': (0, 0, 255),    # Bright Red for Bare Hand / Missing Glove
    'no_helmet': (0, 0, 255),   # Bright Red for Missing Helmet
    'no_shoes': (0, 0, 255),    # Bright Red for Missing Shoes
}

# Global 5-Frame Rolling Smoother History per Track ID
WORKER_HISTORY = {}

def denoise_frame(frame, denoise_h=0, clahe_clip=2.5):
    """
    Apply CLAHE in LAB color space to equalize lamp glare and sharpen hand/glove edges.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def draw_label(img, text, pt, bg_color, text_color=(255, 255, 255), scale=0.5, thickness=1):
    """
    Draws text with a solid filled background card for 100% legibility on factory floors.
    """
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pt
    x = max(5, x)
    y = max(th + 10, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

def extract_frames_smart(video_path, output_dir, diff_threshold=0.2, min_gap_seconds=2, denoise_h=0, clahe_clip=2.5, status_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_gap_frames = int(fps * min_gap_seconds)
    os.makedirs(output_dir, exist_ok=True)

    prev_gray = None
    count, saved, last_saved_frame = 0, 0, -min_gap_frames
    
    print(f"Starting smart frame extraction for: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        should_save = False
        if prev_gray is None:
            should_save = True
        else:
            diff = cv2.absdiff(prev_gray, gray)
            score = diff.mean()
            if score > diff_threshold and (count - last_saved_frame) >= min_gap_frames:
                should_save = True

        if should_save:
            clean_frame = denoise_frame(frame, denoise_h=denoise_h, clahe_clip=clahe_clip)
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved:04d}.jpg"), clean_frame)
            saved += 1
            last_saved_frame = count
            prev_gray = gray

        count += 1
        if status_callback and count % 50 == 0:
            percent = int((count / total_frames) * 100)
            status_callback(percent, saved)

    cap.release()
    return saved

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

def detect_industrial_respirator(head_crop):
    """
    INDUSTRY-READY UNIVERSAL RESPIRATOR DETECTOR:
    Detects factory half-face and full-face elastomeric respirators (3M 6000, 6200, 7500 series):
    1. Yellow/Gold Organic Vapor cartridges (3M 6001/6003)
    2. Pink/Magenta P100 Particulate cartridges (3M 2091/2097)
    Returns: (has_respirator, bounding_box, label_name)
    """
    if head_crop is None or head_crop.size == 0:
        return False, None, ""
        
    hh, hw = head_crop.shape[:2]
    # Check lower 70% of head crop (nose bridge down to below chin)
    y_start = int(hh * 0.25)
    y_end = int(hh * 0.95)
    x_start = int(hw * 0.05)
    x_end = int(hw * 0.95)
    
    face_lower = head_crop[y_start:y_end, x_start:x_end]
    if face_lower.size == 0:
        return False, None, ""
        
    hsv = cv2.cvtColor(face_lower, cv2.COLOR_BGR2HSV)
    
    # Yellow/Gold cartridges (3M 6001/6003 organic vapor)
    yellow_mask = cv2.inRange(hsv, np.array([16, 50, 70]), np.array([36, 255, 255]))
    contours_y, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_boxes = [cv2.boundingRect(cnt) for cnt in contours_y if cv2.contourArea(cnt) > 70]
    
    if y_boxes:
        min_x = min(b[0] for b in y_boxes)
        min_y = min(b[1] for b in y_boxes)
        max_x = max(b[0] + b[2] for b in y_boxes)
        max_y = max(b[1] + b[3] for b in y_boxes)
        pad_x = max(8, int((max_x - min_x) * 0.15))
        pad_y = max(8, int((max_y - min_y) * 0.15))
        box = [
            x_start + max(0, min_x - pad_x),
            y_start + max(0, min_y - pad_y),
            x_start + min(hw, max_x + pad_x),
            y_start + min(hh, max_y + pad_y)
        ]
        return True, box, "respirator"

    # Pink/Magenta cartridges (3M 2091/2097 P100)
    pink_mask = cv2.inRange(hsv, np.array([138, 50, 50]), np.array([170, 255, 255]))
    contours_p, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    p_boxes = [cv2.boundingRect(cnt) for cnt in contours_p if cv2.contourArea(cnt) > 90]
    
    if p_boxes:
        min_x = min(b[0] for b in p_boxes)
        min_y = min(b[1] for b in p_boxes)
        max_x = max(b[0] + b[2] for b in p_boxes)
        max_y = max(b[1] + b[3] for b in p_boxes)
        pad_x = max(8, int((max_x - min_x) * 0.15))
        pad_y = max(8, int((max_y - min_y) * 0.15))
        box = [
            x_start + max(0, min_x - pad_x),
            y_start + max(0, min_y - pad_y),
            x_start + min(hw, max_x + pad_x),
            y_start + min(hh, max_y + pad_y)
        ]
        return True, box, "respirator"

    return False, None, ""

def detect_pink_respirator(head_crop):
    has_resp, box, _ = detect_industrial_respirator(head_crop)
    return has_resp, box

def inspect_worker_fused(frame, box_person, global_detections, ppe_model):
    """
    FUSED MULTI-SCALE INSPECTION:
    Combines global frame YOLO detections, targeted 5x zoom crops,
    and industrial pink respirator spatial verification.
    """
    h, w = frame.shape[:2]
    px1, py1, px2, py2 = map(int, box_person)
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    worker_item_boxes = []
    gear_states = {
        'helmet': 'unknown',
        'mask': 'unknown',
        'glove': 'unknown',
        'shoes': 'unknown'
    }

    # 1. Global Detections matching this worker
    for (box_ppe, cls_id, conf) in global_detections:
        if overlaps(box_ppe, box_person, threshold=0.10) > 0.10:
            label = PPE_CLASSES.get(cls_id, '')
            if not label:
                continue
                
            worker_item_boxes.append((box_ppe, cls_id, conf))
            
            if label in ['mask', 'goggles']:
                gear_states['mask'] = 'present'
            elif label == 'no_mask' and gear_states['mask'] != 'present':
                gear_states['mask'] = 'absent'
            elif label == 'glove':
                gear_states['glove'] = 'present'
            elif label == 'no_glove' and gear_states['glove'] != 'present':
                gear_states['glove'] = 'absent'
            elif label == 'shoes':
                gear_states['shoes'] = 'present'
            elif label == 'no_shoes' and gear_states['shoes'] != 'present':
                gear_states['shoes'] = 'absent'
            elif label == 'helmet':
                gear_states['helmet'] = 'present'
            elif label == 'no_helmet' and gear_states['helmet'] != 'present':
                gear_states['helmet'] = 'absent'

    # 2. Zoomed Head / Face Inspection
    hy1 = max(0, py1 - 10)
    hy2 = min(h, py1 + int(ph * 0.35))
    hx1 = max(0, px1 - 15)
    hx2 = min(w, px2 + 15)
    head_crop = frame[hy1:hy2, hx1:hx2]

    # Verify Pink Respirator in head crop
    has_pink, pink_rect = detect_pink_respirator(head_crop)
    if has_pink and pink_rect:
        gear_states['mask'] = 'present'
        gx1 = hx1 + pink_rect[0]
        gy1 = hy1 + pink_rect[1]
        gx2 = hx1 + pink_rect[2]
        gy2 = hy1 + pink_rect[3]
        worker_item_boxes.append(([gx1, gy1, gx2, gy2], 3, 0.95)) # 3 is 'mask'

    # Zoomed Head Model Prediction
    if head_crop.size > 0 and gear_states['mask'] != 'present':
        h_res = ppe_model.predict(head_crop, conf=0.08, verbose=False, device='cpu')
        if h_res and len(h_res) > 0:
            for box in h_res[0].boxes:
                cls_id = int(box.cls[0].item())
                label = PPE_CLASSES.get(cls_id, '')
                conf = float(box.conf[0].item())
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
                global_box = [hx1 + cx1, hy1 + cy1, hx1 + cx2, hy1 + cy2]
                worker_item_boxes.append((global_box, cls_id, conf))
                
                if label in ['mask', 'goggles']:
                    gear_states['mask'] = 'present'
                elif label == 'no_mask' and gear_states['mask'] != 'present':
                    gear_states['mask'] = 'absent'
                elif label == 'helmet':
                    gear_states['helmet'] = 'present'
                elif label == 'no_helmet':
                    gear_states['helmet'] = 'absent'

    # 3. Zoomed Hand / Arm Inspection
    ay1 = max(0, py1 + int(ph * 0.25))
    ay2 = min(h, py1 + int(ph * 0.85))
    ax1 = max(0, px1 - 30)
    ax2 = min(w, px2 + 30)
    hands_crop = frame[ay1:ay2, ax1:ax2]

    if hands_crop.size > 0:
        a_res = ppe_model.predict(hands_crop, conf=0.08, verbose=False, device='cpu')
        if a_res and len(a_res) > 0:
            for box in a_res[0].boxes:
                cls_id = int(box.cls[0].item())
                label = PPE_CLASSES.get(cls_id, '')
                conf = float(box.conf[0].item())
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
                global_box = [ax1 + cx1, ay1 + cy1, ax1 + cx2, ay1 + cy2]
                worker_item_boxes.append((global_box, cls_id, conf))
                
                if label == 'glove':
                    gear_states['glove'] = 'present'
                elif label == 'no_glove' and gear_states['glove'] != 'present':
                    gear_states['glove'] = 'absent'

    # 4. Zoomed Foot / Boot Inspection
    fy1 = max(0, py1 + int(ph * 0.72))
    fy2 = min(h, py2 + 15)
    fx1 = max(0, px1 - 15)
    fx2 = min(w, px2 + 15)
    feet_crop = frame[fy1:fy2, fx1:fx2]

    if feet_crop.size > 0 and gear_states['shoes'] == 'unknown':
        f_res = ppe_model.predict(feet_crop, conf=0.08, verbose=False, device='cpu')
        if f_res and len(f_res) > 0:
            for box in f_res[0].boxes:
                cls_id = int(box.cls[0].item())
                label = PPE_CLASSES.get(cls_id, '')
                conf = float(box.conf[0].item())
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
                global_box = [fx1 + cx1, fy1 + cy1, fx1 + cx2, fy1 + cy2]
                worker_item_boxes.append((global_box, cls_id, conf))
                
                if label == 'shoes':
                    gear_states['shoes'] = 'present'
                elif label == 'no_shoes':
                    gear_states['shoes'] = 'absent'

    return gear_states, worker_item_boxes

def smooth_worker_compliance(track_id, current_missing):
    """
    ROLLING 5-FRAME TEMPORAL SMOOTHER
    Applies majority voting across 5 consecutive video frames to eliminate single-frame glare flicker.
    """
    if track_id not in WORKER_HISTORY:
        WORKER_HISTORY[track_id] = deque(maxlen=5)

    WORKER_HISTORY[track_id].append(set(current_missing))
    history_len = len(WORKER_HISTORY[track_id])

    smoothed_missing = []
    for item in ['Helmet', 'Mask', 'Gloves', 'Shoes']:
        count = sum(1 for frame_missing in WORKER_HISTORY[track_id] if item in frame_missing)
        if count >= 3 or (history_len < 3 and count == history_len):
            smoothed_missing.append(item)

    return smoothed_missing

def generate_mjpeg_feed(video_path, ppe_model_path, strict_mode=False):
    """
    Generates live MJPEG stream with Fused Multi-Scale Detection,
    spatial respirator validation, and 5-Frame Rolling Smoother.
    """
    person_model = YOLO("yolo11n.pt")
    
    m_model_path = os.path.join(os.path.dirname(__file__), "yolov8m-ppe.pt")
    if os.path.exists(m_model_path):
        ppe_model_path = m_model_path
        
    ppe_model = YOLO(ppe_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
                
        try:
            frame = denoise_frame(frame, denoise_h=0, clahe_clip=2.0)
            
            h, w = frame.shape[:2]
            target_w = 800
            if w > target_w:
                target_h = int(h * (target_w / float(w)))
                frame = cv2.resize(frame, (target_w, target_h))

            # 1. Global PPE Detections on full frame
            global_results = ppe_model.predict(frame, conf=0.08, verbose=False, device='cpu')
            global_detections = []
            if global_results and len(global_results) > 0:
                for box in global_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    global_detections.append((xyxy, cls_id, conf))

            # 2. Multi-Person Tracking
            person_results = person_model.track(frame, persist=True, classes=[0], verbose=False, device='cpu')
            
            if person_results and len(person_results) > 0:
                boxes = person_results[0].boxes
                track_ids = boxes.id.int().tolist() if boxes.id is not None else list(range(1, len(boxes) + 1))
                xyxy_list = boxes.xyxy.tolist()
                
                for idx, (track_id, box_person) in enumerate(zip(track_ids, xyxy_list)):
                    px1, py1, px2, py2 = map(int, box_person)
                    
                    gear_states, item_boxes = inspect_worker_fused(frame, box_person, global_detections, ppe_model)
                    
                    # Draw individual item boxes (Gloves, Mask, Shoes)
                    for (ibox, cls_id, conf) in item_boxes:
                        label = PPE_CLASSES.get(cls_id, '')
                        is_violation = 'no_' in label
                        color = (0, 0, 255) if is_violation else COLOR_PALETTE.get(label, (0, 255, 0))
                        
                        # Only draw shoe box when explicitly detected
                        if label in ['shoes', 'no_shoes'] or label in ['glove', 'no_glove', 'mask', 'no_mask', 'helmet', 'no_helmet', 'goggles']:
                            cv2.rectangle(frame, (int(ibox[0]), int(ibox[1])), (int(ibox[2]), int(ibox[3])), color, 2)
                            draw_label(frame, f"{label.upper()} ({conf:.2f})", (int(ibox[0]), int(ibox[1])), color)

                    raw_missing = []
                    if gear_states['helmet'] == 'absent':
                        raw_missing.append('Helmet')
                    if gear_states['mask'] == 'absent':
                        raw_missing.append('Mask')
                    if gear_states['glove'] == 'absent':
                        raw_missing.append('Gloves')
                    if gear_states['shoes'] == 'absent':
                        raw_missing.append('Shoes')

                    smoothed_missing = smooth_worker_compliance(track_id, raw_missing)
                    
                    is_compliant = len(smoothed_missing) == 0
                    box_color = (0, 220, 0) if is_compliant else (0, 0, 255)
                    bg_color = (0, 140, 0) if is_compliant else (0, 0, 180)
                    
                    cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                    
                    detected_gear = [k.capitalize() for k, v in gear_states.items() if v == 'present']
                    if is_compliant:
                        gear_str = ", ".join(detected_gear) if detected_gear else "OK"
                        status_text = f"Worker #{track_id}: COMPLIANT ({gear_str})"
                    else:
                        status_text = f"Worker #{track_id}: MISSING {', '.join(smoothed_missing)}"
                        
                    label_y = py1 - (idx % 3) * 22
                    draw_label(frame, status_text, (px1, label_y), bg_color)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as ex:
            print("Stream frame error:", ex)
            
        time.sleep(0.03)

    cap.release()

def process_video_pipeline(video_path, output_path, ppe_model_path, snapshots_dir, csv_log_path, strict_mode=False, alert_threshold_frames=5, status_callback=None):
    person_model = YOLO("yolo11n.pt")
    
    m_model_path = os.path.join(os.path.dirname(__file__), "yolov8m-ppe.pt")
    if os.path.exists(m_model_path):
        ppe_model_path = m_model_path
        
    ppe_model = YOLO(ppe_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', ffmpeg_params=['-movflags', '+faststart'])
    
    violation_counters = {}
    active_violations = {}
    violations_history = []
    
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'video_name', 'frame_number', 'video_time_seconds', 'track_id', 'violation_items', 'snapshot_path'])

    frame_num = 0
    print(f"Processing video: {video_path} -> {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        frame = denoise_frame(frame, denoise_h=0, clahe_clip=2.0)
        
        global_results = ppe_model.predict(frame, conf=0.08, verbose=False, device='cpu')
        global_detections = []
        if global_results and len(global_results) > 0:
            for box in global_results[0].boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                global_detections.append((xyxy, cls_id, conf))
        
        person_results = person_model.track(frame, persist=True, classes=[0], verbose=False, device='cpu')
        
        if person_results and len(person_results) > 0 and person_results[0].boxes.id is not None:
            boxes = person_results[0].boxes
            track_ids = boxes.id.int().tolist()
            xyxy_list = boxes.xyxy.tolist()
            
            for idx, (track_id, box_person) in enumerate(zip(track_ids, xyxy_list)):
                gear_states, item_boxes = inspect_worker_fused(frame, box_person, global_detections, ppe_model)
                
                for (ibox, cls_id, conf) in item_boxes:
                    label = PPE_CLASSES.get(cls_id, '')
                    is_violation = 'no_' in label
                    color = (0, 0, 255) if is_violation else COLOR_PALETTE.get(label, (0, 255, 0))
                    if label in ['shoes', 'no_shoes'] or label in ['glove', 'no_glove', 'mask', 'no_mask', 'helmet', 'no_helmet', 'goggles']:
                        cv2.rectangle(frame, (int(ibox[0]), int(ibox[1])), (int(ibox[2]), int(ibox[3])), color, 2)
                        draw_label(frame, f"{label.upper()} ({conf:.2f})", (int(ibox[0]), int(ibox[1])), color)

                raw_missing = []
                if gear_states['helmet'] == 'absent':
                    raw_missing.append('Helmet')
                if gear_states['mask'] == 'absent':
                    raw_missing.append('Mask')
                if gear_states['glove'] == 'absent':
                    raw_missing.append('Gloves')
                if gear_states['shoes'] == 'absent':
                    raw_missing.append('Shoes')

                smoothed_missing = smooth_worker_compliance(track_id, raw_missing)
                
                if track_id not in violation_counters:
                    violation_counters[track_id] = {'Helmet': 0, 'Mask': 0, 'Shoes': 0, 'Gloves': 0}
                    active_violations[track_id] = set()
                    
                current_missing = set(smoothed_missing)
                
                for item in ['Helmet', 'Mask', 'Shoes', 'Gloves']:
                    if item in current_missing:
                        violation_counters[track_id][item] += 1
                    else:
                        violation_counters[track_id][item] = 0
                        if item in active_violations[track_id]:
                            active_violations[track_id].remove(item)
                            
                new_violations = []
                for item, count in violation_counters[track_id].items():
                    if count >= alert_threshold_frames and item not in active_violations[track_id]:
                        new_violations.append(item)
                        active_violations[track_id].add(item)
                
                if new_violations:
                    timestamp_str = datetime.now().isoformat()
                    video_name = os.path.basename(video_path)
                    video_time_sec = round(frame_num / float(fps), 2)
                    
                    with open(csv_log_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        w.writerow([timestamp_str, video_name, frame_num, video_time_sec, track_id, ",".join(new_violations), ""])
                        
                    violations_history.append({
                        "timestamp": timestamp_str,
                        "video_name": video_name,
                        "frame_number": frame_num,
                        "video_time_seconds": video_time_sec,
                        "track_id": track_id,
                        "violation_items": new_violations,
                        "snapshot_path": ""
                    })
                
                is_compliant = len(active_violations[track_id]) == 0
                box_color = (0, 220, 0) if is_compliant else (0, 0, 255)
                bg_color = (0, 140, 0) if is_compliant else (0, 0, 180)
                
                cv2.rectangle(frame, (int(box_person[0]), int(box_person[1])), (int(box_person[2]), int(box_person[3])), box_color, 2)
                
                detected_gear = [k.capitalize() for k, v in gear_states.items() if v == 'present']
                if is_compliant:
                    gear_str = ", ".join(detected_gear) if detected_gear else "OK"
                    status_text = f"Worker #{track_id}: COMPLIANT ({gear_str})"
                else:
                    status_text = f"Worker #{track_id}: MISSING {', '.join(active_violations[track_id])}"
                    
                label_y = int(box_person[1]) - (idx % 3) * 22
                draw_label(frame, status_text, (int(box_person[0]), label_y), bg_color)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
        if status_callback and frame_num % 20 == 0:
            percent = int((frame_num / total_frames) * 100)
            status_callback(percent, len(violations_history))
            
    cap.release()
    writer.close()
    print(f"Finished faststart video processing. Total frames: {frame_num}")
    return violations_history
