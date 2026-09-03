import requests
import time

print("Monitoring annotated video generation...")
for i in range(35):
    try:
        s = requests.get('http://127.0.0.1:8000/api/status').json()
        task = s.get("task", "idle")
        progress = s.get("progress", 0)
        message = s.get("message", "")
        print(f"[{i*3}s] Task: {task} | Progress: {progress}% | {message}")
        if task == "idle" and progress == 100:
            print("Annotated video generated successfully!")
            break
    except Exception as e:
        print("Error:", e)
    time.sleep(3)
