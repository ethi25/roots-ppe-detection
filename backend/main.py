import os
import threading
import shutil
import csv
import time
import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import our utility functions
from utils import extract_frames_smart, process_video_pipeline, generate_mjpeg_feed
from train_pipeline import prepare_dataset_and_train

app = FastAPI(title="PPE Compliance Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted_frames")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_LOG_PATH = os.path.join(BASE_DIR, "violations_log.csv")

# Create directories on startup
for folder in [UPLOADS_DIR, PROCESSED_DIR, SNAPSHOTS_DIR, EXTRACTED_DIR, DATASET_DIR]:
    os.makedirs(folder, exist_ok=True)

# Mount static directories
app.mount("/static/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")
app.mount("/static/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/static/extracted_frames", StaticFiles(directory=EXTRACTED_DIR), name="extracted_frames")

# Global task state tracking
# Used to share progress with the frontend via polling
global_task_state = {
    "task": "idle",       # "idle", "preprocess", "train", "process_video"
    "progress": 0,        # 0 to 100
    "message": "System Idle",
    "logs": []
}

state_lock = threading.Lock()

def update_state(task=None, progress=None, message=None, log_line=None):
    with state_lock:
        if task is not None:
            global_task_state["task"] = task
        if progress is not None:
            global_task_state["progress"] = progress
        if message is not None:
            global_task_state["message"] = message
        if log_line is not None:
            global_task_state["logs"].append(log_line)
            # Keep logs to last 200 lines
            if len(global_task_state["logs"]) > 200:
                global_task_state["logs"].pop(0)

# Request Schemas
class PreprocessRequest(BaseModel):
    video_path: str = "E:\\New folder (2)\\CCTV 1.mp4"
    diff_threshold: float = 0.2
    min_gap_seconds: float = 2.0
    denoise_h: int = 10
    clahe_clip: float = 2.0

class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 4

class ProcessVideoRequest(BaseModel):
    video_path: str = "E:\\New folder (2)\\CCTV 1.mp4"
    strict_mode: bool = True
    alert_threshold_frames: int = 5
    max_frames: int = 500  # For fast demo processing on large videos

# Endpoints
@app.get("/api/stream-video")
def stream_video(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")

@app.get("/api/video-feed")
def video_feed(video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    model_path = os.path.join(BASE_DIR, "yolov8m-ppe.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "yolov8n-ppe.pt")
                
    return StreamingResponse(
        generate_mjpeg_feed(video_path, model_path, strict_mode=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/status")
def get_status():
    """Returns the current background task status and logs."""
    return global_task_state

@app.get("/api/violations")
def get_violations():
    """Reads violations from the CSV log and returns them."""
    violations = []
    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                violations.append({
                    "timestamp": row.get("timestamp"),
                    "video_name": row.get("video_name"),
                    "frame_number": int(row.get("frame_number", 0)),
                    "track_id": int(row.get("track_id", 0)),
                    "violation_items": row.get("violation_items", "").split(","),
                    "snapshot_path": row.get("snapshot_path")
                })
    return violations

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handles uploading a new video file."""
    try:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "video_path": file_path, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/extracted-frames")
def list_extracted_frames():
    """Lists the paths of all extracted frames currently stored."""
    frames = []
    if os.path.exists(EXTRACTED_DIR):
        files = [f for f in os.listdir(EXTRACTED_DIR) if f.lower().endswith(('.jpg', '.png'))]
        # Sort files numerically
        try:
            files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
        except Exception:
            files.sort()
        frames = [f"/static/extracted_frames/{f}" for f in files]
    return frames

@app.get("/api/processed-videos")
def list_processed_videos():
    """Lists all processed annotated videos."""
    videos = []
    if os.path.exists(PROCESSED_DIR):
        files = [f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith('.mp4')]
        videos = [f"/static/processed/{f}" for f in files]
    return videos

# Background execution functions
def run_preprocess_bg(req: PreprocessRequest):
    update_state(task="preprocess", progress=0, message="Starting frame extraction...")
    try:
        # Clear previous frames
        for f in os.listdir(EXTRACTED_DIR):
            fpath = os.path.join(EXTRACTED_DIR, f)
            if os.path.isfile(fpath):
                os.remove(fpath)

        def cb(percent, saved_count):
            msg = f"Extracted {saved_count} frames... {percent}% complete"
            update_state(progress=percent, message=msg, log_line=msg)

        saved = extract_frames_smart(
            video_path=req.video_path,
            output_dir=EXTRACTED_DIR,
            diff_threshold=req.diff_threshold,
            min_gap_seconds=req.min_gap_seconds,
            denoise_h=req.denoise_h,
            clahe_clip=req.clahe_clip,
            status_callback=cb
        )
        update_state(task="idle", progress=100, message=f"Success! Extracted {saved} frames.", log_line=f"Preprocessing finished. Extracted {saved} frames.")
    except Exception as e:
        err_msg = f"Error in preprocessing: {str(e)}"
        update_state(task="idle", progress=0, message=err_msg, log_line=err_msg)

def run_train_bg(req: TrainRequest):
    update_state(task="train", progress=0, message="Initializing training database...")
    try:
        # Check if we have extracted frames first
        frames = [f for f in os.listdir(EXTRACTED_DIR) if f.lower().endswith(('.jpg', '.png'))]
        if not frames:
            # Run preprocessor automatically with default settings if empty
            update_state(message="No extracted frames found. Running frame extractor first...")
            extract_frames_smart(
                video_path="E:\\New folder (2)\\CCTV 1.mp4",
                output_dir=EXTRACTED_DIR,
                diff_threshold=25,
                min_gap_seconds=2.0
            )

        # Base PPE model path
        base_model_path = "C:\\Users\\Ethirajan\\.gemini\\antigravity-ide\scratch\\yolov8n-ppe.pt"
        if not os.path.exists(base_model_path):
            base_model_path = os.path.join(BASE_DIR, "..", "yolov8n-ppe.pt")
            if not os.path.exists(base_model_path):
                # Fallback to downloading
                import urllib.request
                update_state(message="Downloading base PPE model weights...")
                url = 'https://huggingface.co/keremberke/yolov8n-protective-equipment-detection/resolve/main/best.pt'
                urllib.request.urlretrieve(url, base_model_path)

        def cb(msg, percent):
            update_state(progress=percent, message=msg, log_line=msg)

        best_weights = prepare_dataset_and_train(
            extracted_dir=EXTRACTED_DIR,
            dataset_dir=DATASET_DIR,
            model_path=base_model_path,
            epochs=req.epochs,
            batch_size=req.batch_size,
            status_callback=cb
        )
        
        # Save model path details
        with open(os.path.join(BASE_DIR, "latest_weights.txt"), "w") as f:
            f.write(best_weights)

        update_state(task="idle", progress=100, message="Training completed!", log_line=f"Training complete. Weights at: {best_weights}")
    except Exception as e:
        err_msg = f"Error in training: {str(e)}"
        update_state(task="idle", progress=0, message=err_msg, log_line=err_msg)

def run_process_video_bg(req: ProcessVideoRequest):
    update_state(task="process_video", progress=0, message="Initializing video inference and tracking...")
    try:
        # Load weights
        weights_file = os.path.join(BASE_DIR, "latest_weights.txt")
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                model_path = f.read().strip()
        else:
            model_path = "C:\\Users\\Ethirajan\\.gemini\\antigravity-ide\\scratch\\yolov8n-ppe.pt"
            if not os.path.exists(model_path):
                model_path = os.path.join(BASE_DIR, "..", "yolov8n-ppe.pt")

        output_filename = f"processed_{int(time.time())}_{os.path.basename(req.video_path)}"
        # Sanitize filename (replace spaces)
        output_filename = output_filename.replace(" ", "_")
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # Temporary trim if max_frames is set to avoid 1GB video rendering taking forever
        video_source = req.video_path
        if req.max_frames > 0:
            update_state(message="Preparing temporary trimmed video for fast demo...")
            cap = cv2.VideoCapture(req.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            temp_trimmed_path = os.path.join(UPLOADS_DIR, f"temp_trimmed_{int(time.time())}.mp4")
            out_trim = cv2.VideoWriter(temp_trimmed_path, fourcc, fps, (width, height))
            
            count = 0
            while count < req.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out_trim.write(frame)
                count += 1
            cap.release()
            out_trim.release()
            video_source = temp_trimmed_path

        def cb(percent, violations_count):
            msg = f"Processed {percent}% of video frames. Active violations: {violations_count}"
            update_state(progress=percent, message=msg, log_line=msg)

        violations = process_video_pipeline(
            video_path=video_source,
            output_path=output_path,
            ppe_model_path=model_path,
            snapshots_dir=SNAPSHOTS_DIR,
            csv_log_path=CSV_LOG_PATH,
            strict_mode=req.strict_mode,
            alert_threshold_frames=req.alert_threshold_frames,
            status_callback=cb
        )

        # Remove temporary trimmed video
        if req.max_frames > 0 and os.path.exists(video_source):
            try:
                os.remove(video_source)
            except Exception:
                pass

        update_state(task="idle", progress=100, message=f"Video processed successfully! Found {len(violations)} violations.", log_line="Inference completed.")
    except Exception as e:
        err_msg = f"Error processing video: {str(e)}"
        update_state(task="idle", progress=0, message=err_msg, log_line=err_msg)

# Triggers
@app.post("/api/preprocess")
def preprocess_video(req: PreprocessRequest, background_tasks: BackgroundTasks):
    """Triggers smart frame extraction in a background thread."""
    if global_task_state["task"] != "idle":
        raise HTTPException(status_code=400, detail="Another background task is currently running.")
    background_tasks.add_task(run_preprocess_bg, req)
    return {"status": "started", "message": "Frame extraction started in background."}

@app.post("/api/train")
def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """Triggers YOLO fine-tuning on the extracted frames."""
    if global_task_state["task"] != "idle":
        raise HTTPException(status_code=400, detail="Another background task is currently running.")
    background_tasks.add_task(run_train_bg, req)
    return {"status": "started", "message": "Model training started in background."}

@app.post("/api/process-video")
def process_video(req: ProcessVideoRequest, background_tasks: BackgroundTasks):
    """Triggers the full tracking and compliance verification pipeline on a video."""
    if global_task_state["task"] != "idle":
        raise HTTPException(status_code=400, detail="Another background task is currently running.")
    background_tasks.add_task(run_process_video_bg, req)
    return {"status": "started", "message": "Video processing started in background."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
