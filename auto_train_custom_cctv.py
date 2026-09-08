import os
import cv2
import glob
import shutil
import random
import yaml
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")

# Primary CCTV Video Paths - ONLY real factory CCTV footage
VIDEO_PATHS = [
    r"E:\New folder (2)\CCTV 1.mp4",
    r"E:\New folder (2)\CCTV 2.mp4"
]

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

def denoise_frame(frame, clahe_clip=2.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def detect_pink_respirator(head_crop):
    """
    Detects industrial pink dual-cartridge respirators using HSV color-spatial analysis.
    """
    if head_crop is None or head_crop.size == 0:
        return False, None
        
    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([135, 45, 50]), np.array([175, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 45, 50]), np.array([12, 255, 255]))
    pink_mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            return True, [x, y, x + w, y + h]
            
    return False, None

def extract_and_auto_annotate(max_frames_per_video=40):
    print(f"Step 1: Extracting pure CCTV frames from {len(VIDEO_PATHS)} factory cameras...")
    
    extracted_img_dir = os.path.join(DATASET_DIR, "raw_images")
    extracted_lbl_dir = os.path.join(DATASET_DIR, "raw_labels")
    
    if os.path.exists(DATASET_DIR):
        try:
            shutil.rmtree(DATASET_DIR)
        except Exception as e:
            print("Cleanup note:", e)
            
    os.makedirs(extracted_img_dir, exist_ok=True)
    os.makedirs(extracted_lbl_dir, exist_ok=True)

    model_path = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(BACKEND_DIR, "yolov8n-ppe.pt")
    auto_model = YOLO(model_path)
    person_model = YOLO("yolo11n.pt")

    saved_count = 0

    for vid_idx, vid_path in enumerate(VIDEO_PATHS):
        if not os.path.exists(vid_path):
            print(f"Skipping (not found): {vid_path}")
            continue
            
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Sample every 2.5 seconds (60 frames) sequentially to maintain perfect H.265 GOP decoding
        sample_step = int(fps * 2.5)
        
        print(f"[{vid_idx+1}/{len(VIDEO_PATHS)}] Sequential stream read: {os.path.basename(vid_path)} (every {sample_step} frames)")

        vid_saved = 0
        curr_frame = 0

        while vid_saved < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if curr_frame % sample_step == 0:
                clean = denoise_frame(frame, clahe_clip=2.0)
                h, w = clean.shape[:2]

                # Check if person is present in the factory frame
                person_res = person_model.predict(clean, conf=0.30, classes=[0], verbose=False, device='cpu')
                if person_res and len(person_res) > 0 and len(person_res[0].boxes) > 0:
                    img_name = f"cctv{vid_idx+1}_frame_{saved_count:04d}.jpg"
                    img_path = os.path.join(extracted_img_dir, img_name)
                    cv2.imwrite(img_path, clean)

                    label_lines = []
                    
                    # PPE prediction
                    results = auto_model.predict(clean, conf=0.12, verbose=False, device='cpu')
                    if results and len(results) > 0:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0].item())
                            xywhn = box.xywhn[0].tolist()
                            label_lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

                    # Pink respirator spatial verification
                    for pbox in person_res[0].boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)
                        
                        hy1 = max(0, py1 - 10)
                        hy2 = min(h, py1 + int(ph * 0.35))
                        hx1 = max(0, px1 - 15)
                        hx2 = min(w, px2 + 15)
                        head_crop = clean[hy1:hy2, hx1:hx2]
                        
                        has_pink, pink_box = detect_pink_respirator(head_crop)
                        if has_pink and pink_box:
                            gx1 = hx1 + pink_box[0]
                            gy1 = hy1 + pink_box[1]
                            gx2 = hx1 + pink_box[2]
                            gy2 = hy1 + pink_box[3]
                            
                            cx = ((gx1 + gx2) / 2.0) / float(w)
                            cy = ((gy1 + gy2) / 2.0) / float(h)
                            bw = (gx2 - gx1) / float(w)
                            bh = (gy2 - gy1) / float(h)
                            label_lines.append(f"3 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}") # 3 is 'mask'

                    txt_name = f"cctv{vid_idx+1}_frame_{saved_count:04d}.txt"
                    txt_path = os.path.join(extracted_lbl_dir, txt_name)
                    with open(txt_path, 'w') as f:
                        f.write("\n".join(label_lines))

                    saved_count += 1
                    vid_saved += 1

            curr_frame += 1

        cap.release()

    print(f"Extracted and auto-annotated {saved_count} pure CCTV frames.")
    return saved_count, extracted_img_dir, extracted_lbl_dir

def prepare_yolo_dataset(img_dir, lbl_dir):
    print("Step 2: Preparing clean train/val splits with pure CCTV frames...")
    
    train_img = os.path.join(DATASET_DIR, "images", "train")
    val_img = os.path.join(DATASET_DIR, "images", "val")
    train_lbl = os.path.join(DATASET_DIR, "labels", "train")
    val_lbl = os.path.join(DATASET_DIR, "labels", "val")
    
    for d in [train_img, val_img, train_lbl, val_lbl]:
        os.makedirs(d, exist_ok=True)

    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    random.shuffle(images)
    
    val_size = max(2, int(len(images) * 0.15))
    val_images = set(images[:val_size])
    
    for img_path in images:
        base_name = os.path.basename(img_path)
        txt_name = base_name.replace(".jpg", ".txt")
        txt_path = os.path.join(lbl_dir, txt_name)
        
        target_img_dir = val_img if img_path in val_images else train_img
        target_lbl_dir = val_lbl if img_path in val_images else train_lbl
        
        shutil.copy(img_path, os.path.join(target_img_dir, base_name))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(target_lbl_dir, txt_name))

    yaml_content = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': PPE_CLASSES
    }
    yaml_path = os.path.join(DATASET_DIR, "ppe_custom.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"Clean Pure CCTV Dataset Ready: {len(images) - val_size} train frames, {val_size} val frames.")
    return yaml_path

def run_fine_tuning(yaml_path):
    print("Step 3: Launching Pure CCTV YOLOv8m Fine-Tuning (20 Epochs)...")
    
    base_weights = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    model = YOLO(base_weights)

    project_dir = os.path.join(BACKEND_DIR, "runs")
    
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=800,
        batch=4,
        workers=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        scale=0.5,
        project=project_dir,
        name="factory_cctv_pure_run",
        exist_ok=True,
        device="cpu"
    )

    best_weights = os.path.join(project_dir, "factory_cctv_pure_run", "weights", "best.pt")
    target_weights = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        weights_txt = os.path.join(BACKEND_DIR, "latest_weights.txt")
        with open(weights_txt, 'w') as f:
            f.write(target_weights)
        print(f"SUCCESS! Pure CCTV-trained weights deployed to: {target_weights}")

if __name__ == "__main__":
    count, img_dir, lbl_dir = extract_and_auto_annotate(max_frames_per_video=40)
    if count > 0:
        yaml_path = prepare_yolo_dataset(img_dir, lbl_dir)
        run_fine_tuning(yaml_path)
    else:
        print("Error: No frames extracted.")
