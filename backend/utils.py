import cv2
import numpy as np
import os
import time
import csv
import imageio
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
    'mask': (255, 255, 0),      # Cyan for Present Mask
    'glove': (0, 255, 255),     # Bright Yellow for Present Glove
    'helmet': (255, 191, 0),    # Blue for Present Helmet
    'shoes': (255, 0, 255),     # Magenta for Present Shoes
    'goggles': (255, 255, 0),   # Cyan for Respirator/Goggles
    'no_mask': (0, 0, 255),     # Bright Red for Bare Face / Missing Mask
    'no_glove': (0, 0, 255),    # Bright Red for Bare Hand / Missing Glove
    'no_helmet': (0, 0, 255),   # Bright Red for Missing Helmet
    'no_shoes': (0, 0, 255),    # Bright Red for Missing Shoes
}

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

def overlaps(box_ppe, box_person, threshold=0.08):
    x1_p, y1_p, x2_p, y2_p = box_person
    x1_e, y1_e, x2_e, y2_e = box_ppe
    
    x1_i = max(x1_p, x1_e)
    y1_i = max(y1_p, y1_e)
    x2_i = min(x2_p, x2_e)
    y2_i = min(y2_p, y2_e)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
        
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    ppe_area = (x2_e - x1_e) * (y2_e - y1_e)
    
    return inter_area / float(ppe_area) if ppe_area > 0 else 0.0

def check_compliance(person_box, ppe_detections, strict_mode=False):
    states = {
        'helmet': 'unknown',
        'mask': 'unknown',
        'shoes': 'unknown',
        'glove': 'unknown'
    }
    
    for box_ppe, class_id, conf in ppe_detections:
        label = PPE_CLASSES.get(class_id, '')
        if not label:
            continue
            
        if overlaps(box_ppe, person_box, threshold=0.08) > 0.08:
            if label == 'helmet':
                states['helmet'] = 'present'
            elif label == 'no_helmet':
                states['helmet'] = 'absent'
            elif label in ['mask', 'goggles']:
                states['mask'] = 'present'
            elif label == 'no_mask':
                states['mask'] = 'absent'
            elif label == 'shoes':
                states['shoes'] = 'present'
            elif label == 'no_shoes':
                states['shoes'] = 'absent'
            elif label == 'glove':
                states['glove'] = 'present'
            elif label == 'no_glove':
                states['glove'] = 'absent'

    missing_items = []
    
    # Flag missing if explicitly detected as absent or missing
    if states['helmet'] == 'absent':
        missing_items.append('Helmet')
    if states['mask'] == 'absent':
        missing_items.append('Mask')
    if states['glove'] == 'absent':
        missing_items.append('Gloves')
    if states['shoes'] == 'absent':
        missing_items.append('Shoes')
        
    return missing_items, states

def generate_mjpeg_feed(video_path, ppe_model_path, strict_mode=False):
    """
    Generates live MJPEG stream with separate person tracking, CLAHE glare reduction,
    and individual per-hand glove, mask, and shoe bounding boxes.
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
            # Glare equalization for factory lighting
            frame = denoise_frame(frame, denoise_h=0, clahe_clip=2.0)
            
            h, w = frame.shape[:2]
            target_w = 800
            if w > target_w:
                target_h = int(h * (target_w / float(w)))
                frame = cv2.resize(frame, (target_w, target_h))

            # Multi-person tracking and PPE item detection
            person_results = person_model.track(frame, persist=True, classes=[0], verbose=False, device='cpu')
            ppe_results = ppe_model.predict(frame, conf=0.10, verbose=False, device='cpu')
            
            ppe_detections = []
            if ppe_results and len(ppe_results) > 0:
                for box in ppe_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    ppe_detections.append((xyxy, cls_id, conf))
                    
                    label = PPE_CLASSES.get(cls_id, '')
                    is_violation = 'no_' in label
                    color = (0, 0, 255) if is_violation else COLOR_PALETTE.get(label, (0, 255, 0))
                    
                    # Draw SEPARATE individual bounding box around detected item (gloves, mask, shoes, etc.)
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    draw_label(frame, f"{label.upper()} ({conf:.2f})", (int(xyxy[0]), int(xyxy[1])), color)

            if person_results and len(person_results) > 0:
                boxes = person_results[0].boxes
                track_ids = boxes.id.int().tolist() if boxes.id is not None else list(range(1, len(boxes) + 1))
                xyxy_list = boxes.xyxy.tolist()
                
                for idx, (track_id, box_person) in enumerate(zip(track_ids, xyxy_list)):
                    px1, py1, px2, py2 = map(int, box_person)
                    
                    missing, states = check_compliance(box_person, ppe_detections, strict_mode=strict_mode)
                    
                    is_compliant = len(missing) == 0
                    box_color = (0, 220, 0) if is_compliant else (0, 0, 255)
                    bg_color = (0, 140, 0) if is_compliant else (0, 0, 180)
                    
                    # Person overall bounding box (Red if missing gear, Green if compliant)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                    
                    detected_gear = [k.capitalize() for k, v in states.items() if v == 'present']
                    if is_compliant:
                        gear_str = ", ".join(detected_gear) if detected_gear else "OK"
                        status_text = f"Worker #{track_id}: COMPLIANT ({gear_str})"
                    else:
                        status_text = f"Worker #{track_id}: MISSING {', '.join(missing)}"
                        
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
        
        person_results = person_model.track(frame, persist=True, classes=[0], verbose=False, device='cpu')
        ppe_results = ppe_model.predict(frame, conf=0.10, verbose=False, device='cpu')
        
        ppe_detections = []
        if ppe_results and len(ppe_results) > 0:
            for box in ppe_results[0].boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                ppe_detections.append((xyxy, cls_id, conf))
                
                label = PPE_CLASSES.get(cls_id, '')
                is_violation_class = 'no_' in label
                color = (0, 0, 255) if is_violation_class else COLOR_PALETTE.get(label, (0, 255, 0))
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                draw_label(frame, f"{label.upper()} ({conf:.2f})", (int(xyxy[0]), int(xyxy[1])), color)

        if person_results and len(person_results) > 0 and person_results[0].boxes.id is not None:
            boxes = person_results[0].boxes
            track_ids = boxes.id.int().tolist()
            xyxy_list = boxes.xyxy.tolist()
            
            for idx, (track_id, box_person) in enumerate(zip(track_ids, xyxy_list)):
                missing, states = check_compliance(box_person, ppe_detections, strict_mode=strict_mode)
                
                if track_id not in violation_counters:
                    violation_counters[track_id] = {'Helmet': 0, 'Mask': 0, 'Shoes': 0, 'Gloves': 0}
                    active_violations[track_id] = set()
                    
                current_missing = set(missing)
                
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
                
                detected_gear = [k.capitalize() for k, v in states.items() if v == 'present']
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
