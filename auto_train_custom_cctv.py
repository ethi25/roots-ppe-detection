import os
import cv2
import glob
import shutil
import random
import yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")

VIDEOS_DIR = r"C:\Users\Ethirajan\Videos"
VIDEO_PATHS = []

if os.path.exists(VIDEOS_DIR):
    for f in os.listdir(VIDEOS_DIR):
        if f.endswith(".mp4") and "training" in f:
            VIDEO_PATHS.append(os.path.join(VIDEOS_DIR, f))

if not VIDEO_PATHS:
    VIDEO_PATHS = [
        r"C:\Users\Ethirajan\Videos\training(1).mp4",
        r"C:\Users\Ethirajan\Videos\training(2).mp4"
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

def denoise_frame(frame, clahe_clip=2.5):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def extract_and_auto_annotate():
    print(f"Step 1: Extracting & auto-annotating frames from {len(VIDEO_PATHS)} CCTV clips...")
    
    extracted_img_dir = os.path.join(DATASET_DIR, "raw_images")
    extracted_lbl_dir = os.path.join(DATASET_DIR, "raw_labels")
    
    # Remove old dataset directory & cache files cleanly
    if os.path.exists(DATASET_DIR):
        try:
            shutil.rmtree(DATASET_DIR)
        except Exception as e:
            print("Cleanup info:", e)
            
    os.makedirs(extracted_img_dir, exist_ok=True)
    os.makedirs(extracted_lbl_dir, exist_ok=True)

    model_path = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(BACKEND_DIR, "yolov8n-ppe.pt")
    auto_model = YOLO(model_path)

    saved_count = 0

    for vid_idx, vid_path in enumerate(VIDEO_PATHS):
        if not os.path.exists(vid_path):
            continue
            
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = int(fps * 2.0)
        
        print(f"[{vid_idx+1}/{len(VIDEO_PATHS)}] Extracting from: {os.path.basename(vid_path)}")

        curr_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if curr_frame % frame_step == 0:
                clean = denoise_frame(frame, clahe_clip=2.5)
                img_name = f"cctv{vid_idx+1}_frame_{saved_count:04d}.jpg"
                img_path = os.path.join(extracted_img_dir, img_name)
                cv2.imwrite(img_path, clean)

                results = auto_model.predict(clean, conf=0.10, verbose=False, device='cpu')
                
                label_lines = []
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        xywhn = box.xywhn[0].tolist()
                        label_lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

                txt_name = f"cctv{vid_idx+1}_frame_{saved_count:04d}.txt"
                txt_path = os.path.join(extracted_lbl_dir, txt_name)
                with open(txt_path, 'w') as f:
                    f.write("\n".join(label_lines))

                saved_count += 1
            curr_frame += 1

        cap.release()

    print(f"Extracted and auto-annotated {saved_count} clean frames.")
    return saved_count, extracted_img_dir, extracted_lbl_dir

def prepare_yolo_dataset(img_dir, lbl_dir):
    print("Step 2: Preparing clean train/val dataset splits...")
    
    train_img = os.path.join(DATASET_DIR, "images", "train")
    val_img = os.path.join(DATASET_DIR, "images", "val")
    train_lbl = os.path.join(DATASET_DIR, "labels", "train")
    val_lbl = os.path.join(DATASET_DIR, "labels", "val")
    
    for d in [train_img, val_img, train_lbl, val_lbl]:
        os.makedirs(d, exist_ok=True)

    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    random.shuffle(images)
    
    val_size = max(1, int(len(images) * 0.2))
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

    print(f"Dataset ready: {len(images) - val_size} train images, {val_size} val images.")
    return yaml_path

def run_fine_tuning(yaml_path):
    print("Step 3: Launching 20-Epoch High-Precision YOLOv8m Fine-Tuning...")
    
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
        name="factory_clean_run",
        exist_ok=True,
        device="cpu"
    )

    best_weights = os.path.join(project_dir, "factory_clean_run", "weights", "best.pt")
    target_weights = os.path.join(BACKEND_DIR, "yolov8m-ppe.pt")
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        weights_txt = os.path.join(BACKEND_DIR, "latest_weights.txt")
        with open(weights_txt, 'w') as f:
            f.write(target_weights)
        print(f"SUCCESS! Fine-tuned weights updated at: {target_weights}")

if __name__ == "__main__":
    count, img_dir, lbl_dir = extract_and_auto_annotate()
    if count > 0:
        yaml_path = prepare_yolo_dataset(img_dir, lbl_dir)
        run_fine_tuning(yaml_path)
    else:
        print("Error: No frames extracted.")
