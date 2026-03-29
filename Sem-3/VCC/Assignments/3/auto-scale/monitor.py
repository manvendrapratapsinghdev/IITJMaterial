import psutil
import time
import subprocess

THRESHOLD = 75

def trigger_cloud():
    print("Threshold exceeded. Scaling to cloud...")
    subprocess.run(["bash", "deploy_cloud.sh"])

while True:
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent

    print(f"CPU: {cpu}%, Memory: {memory}%")

    if cpu > THRESHOLD or memory > THRESHOLD:
        trigger_cloud()
        break

    time.sleep(5)