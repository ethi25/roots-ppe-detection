import requests
import time

print("Monitoring video inference progress...")
for i in range(40):
    try:
        s = requests.get('http://127.0.0.1:8000/api/status').json()
        task = s.get("task", "idle")
        progress = s.get("progress", 0)
        message = s.get("message", "")
        print(f"[{i*3}s] Task: {task} | Progress: {progress}% | {message}")
        if task == "idle" and progress == 100:
            print("Video processing finished successfully!")
            break
    except Exception as e:
        print("Error:", e)
    time.sleep(3)
