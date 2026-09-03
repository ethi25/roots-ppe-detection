import requests, os, time

d = 'C:/Users/Ethirajan/.gemini/antigravity-ide/scratch/ppe-compliance-detector/backend/extracted_frames'
print('Monitoring extraction progress...')

for i in range(8):
    time.sleep(30)
    try:
        s = requests.get('http://127.0.0.1:8000/api/status').json()
        n = len([f for f in os.listdir(d) if f.endswith('.jpg')])
        elapsed = (i + 1) * 30
        msg = s.get('message', '')
        task = s.get('task', '')
        progress = s.get('progress', 0)
        print(f"[{elapsed}s] task={task} | progress={progress}% | frames_on_disk={n} | msg={msg}")
        if task == 'idle' and n > 1:
            print('Extraction DONE!')
            break
    except Exception as ex:
        print(f'Error: {ex}')
