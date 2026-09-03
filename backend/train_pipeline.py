import os
import shutil
import random
from ultralytics import YOLO

def prepare_dataset_and_train(extracted_dir, dataset_dir, model_path, epochs=5, batch_size=4, status_callback=None):
    """
    1. Splits extracted images into Train and Val sets.
    2. Runs auto-labeling using the pre-trained PPE model.
    3. Generates the dataset YAML file.
    4. Triggers YOLO fine-tuning.
    """
    if status_callback:
        status_callback("Preparing dataset directories...", 5)

    # 1. Clean and create dataset directories
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

    # List all extracted images
    all_images = [f for f in os.listdir(extracted_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        raise ValueError("No extracted frames found in the source directory.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    if status_callback:
        status_callback(f"Loading auto-labeler model from {model_path}...", 15)

    # Load the auto-labeler model
    model = YOLO(model_path)
    
    # Define mapping helper
    def process_split(images, split_name):
        total = len(images)
        for idx, img_name in enumerate(images):
            src_img_path = os.path.join(extracted_dir, img_name)
            dst_img_path = os.path.join(dataset_dir, split_name, 'images', img_name)
            
            # Copy image
            shutil.copy(src_img_path, dst_img_path)
            
            # Predict and write label
            results = model.predict(src_img_path, conf=0.35, verbose=False)
            boxes = results[0].boxes
            
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(dataset_dir, split_name, 'labels', label_name)
            
            with open(label_path, 'w') as f:
                for box in boxes:
                    cls = int(box.cls[0].item())
                    # xywhn gives normalized coordinates
                    xywhn = box.xywhn[0].tolist()
                    f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                    
            if status_callback and idx % max(1, total // 5) == 0:
                progress = 20 + int((idx / total) * 30) if split_name == 'train' else 50 + int((idx / total) * 10)
                status_callback(f"Auto-labeling {split_name} split: {idx}/{total}...", progress)

    # Run auto-labeling
    process_split(train_images, 'train')
    process_split(val_images, 'val')

    if status_callback:
        status_callback("Generating ppe_data.yaml configuration...", 65)

    # 3. Create YAML file (Absolute paths for safety)
    yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: train/images
val: val/images

nc: 10
names: ['glove', 'goggles', 'helmet', 'mask', 'no_glove', 'no_goggles', 'no_helmet', 'no_mask', 'no_shoes', 'shoes']
"""
    yaml_path = os.path.join(dataset_dir, 'ppe_data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    if status_callback:
        status_callback("Starting custom fine-tuning...", 70)

    # 4. Train Model
    # Start from pretrained weights
    train_model = YOLO(model_path)
    
    # Custom training callback to capture progress logs
    class ProgressLogger:
        def on_train_epoch_end(self, trainer):
            epoch = trainer.epoch + 1
            epochs = trainer.epochs
            loss = trainer.loss.item() if hasattr(trainer, 'loss') else 0.0
            # For logging mAP
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            map50 = metrics.get('metrics/mAP50(B)', 0.0)
            
            log_str = f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - mAP50: {map50:.4f}"
            print(f"[YOLO_TRAIN_LOG] {log_str}")
            if status_callback:
                progress = 70 + int((epoch / epochs) * 25)
                status_callback(log_str, progress)

    logger = ProgressLogger()
    train_model.add_callback('on_train_epoch_end', logger.on_train_epoch_end)

    # Train
    results = train_model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device='cpu', # default to CPU since CUDA is not installed for PyTorch on this PC
        project=os.path.join(dataset_dir, 'runs'),
        name='ppe_train',
        verbose=True
    )

    if status_callback:
        status_callback("Fine-tuning completed successfully!", 100)

    # Path to best weights
    best_weights_path = os.path.join(dataset_dir, 'runs', 'ppe_train', 'weights', 'best.pt')
    if not os.path.exists(best_weights_path):
        # Fallback to model_path if best.pt is not found for some reason
        print(f"Warning: best.pt not found, returning initial weights: {model_path}")
        return model_path

    print(f"Fine-tuned weights saved to: {best_weights_path}")
    return best_weights_path
