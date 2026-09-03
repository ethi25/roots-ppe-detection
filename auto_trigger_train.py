"""
Auto-trigger script: Monitors extraction, then auto-triggers training
once enough frames are available.
"""
import requests, os, time

FRAMES_DIR = 'C:/Users/Ethirajan/.gemini/antigravity-ide/scratch/ppe-compliance-detector/backend/extracted_frames'
API = 'http://127.0.0.1:8000'
MIN_FRAMES_TO_TRAIN = 80  # Trigger training as soon as 80 frames are ready
TRAIN_EPOCHS = 5
TRAIN_BATCH = 4

print('=== Auto-trigger watcher started ===')
print(f'Will trigger training when >= {MIN_FRAMES_TO_TRAIN} frames extracted')

training_triggered = False

while True:
    time.sleep(20)
    try:
        s = requests.get(f'{API}/api/status', timeout=3).json()
        n = len([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
        task = s.get('task', 'idle')
        progress = s.get('progress', 0)
        msg = s.get('message', '')
        print(f'[{time.strftime("%H:%M:%S")}] task={task} | progress={progress}% | frames={n}')

        # If extraction is done or we have enough frames and training hasn't started
        if not training_triggered and task == 'idle' and n >= MIN_FRAMES_TO_TRAIN:
            print(f'\n>>> {n} frames available. Triggering training for {TRAIN_EPOCHS} epochs...')
            r = requests.post(f'{API}/api/train', json={
                'epochs': TRAIN_EPOCHS,
                'batch_size': TRAIN_BATCH
            })
            print('Train response:', r.status_code, r.json())
            training_triggered = True

        # If training is running, show training logs
        if task == 'train':
            logs = s.get('logs', [])
            if logs:
                print('  Latest log:', logs[-1])

        # Exit once training is complete
        if training_triggered and task == 'idle' and progress == 100:
            print('\n=== Training COMPLETE! Model ready. ===')
            # Check for best weights
            weights_file = 'C:/Users/Ethirajan/.gemini/antigravity-ide/scratch/ppe-compliance-detector/backend/latest_weights.txt'
            if os.path.exists(weights_file):
                with open(weights_file) as f:
                    print('Best weights path:', f.read().strip())
            break

    except Exception as ex:
        print(f'Error: {ex}')
        time.sleep(5)
