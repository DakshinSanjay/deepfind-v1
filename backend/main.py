import os
import subprocess
import platform
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

# 1. Define the expected payload structure
class RevealPayload(BaseModel):
    file_path: str

@app.get("/api/search")
def search(q: str):
    return [{
        "file_name": "receipt_photo_final2.jpg",
        # Use a path that actually exists on your Linux machine to test it, like your Documents folder
        "file_path": "/home/dakshinsanjay/Desktop/Karthik/deepfind", 
        "file_type": "image",
        "snippet": "…acknowledgement of fee payment received on 12 March…",
        "score": 0.91,
        "match_type": "semantic"
    }]

# 2. Add the Reveal endpoint
@app.post("/api/reveal")
def reveal_file(payload: RevealPayload):
    path = os.path.normpath(payload.file_path)
    
    # Identify the OS and run the native file manager command
    system = platform.system()
    if system == "Windows":
        subprocess.run(f'explorer /select,"{path}"')
    elif system == "Darwin":
        subprocess.run(["open", "-R", path])
    else:
        # Linux (your current environment)
        dir_path = os.path.dirname(path) if os.path.isfile(path) else path
        subprocess.run(["xdg-open", dir_path])
        
    return {"status": "ok"}

# Add this at the bottom of backend/main.py
@app.get("/api/status")
def get_status():
    # In the final version, HP's module will provide these real numbers using psutil
    return {
        "indexed_files": 1420,
        "tier2_queue": 3,
        "cpu_percent": 12.4,
        "ram_percent": 48.1,
        "on_battery": False,
        "is_paused": False
    }

# Add this at the bottom of backend/main.py
class SettingsPayload(BaseModel):
    watched_paths: list[str]
    cpu_limit: int
    ram_limit: int
    pause_on_battery: bool

@app.post("/api/settings")
def update_settings(settings: SettingsPayload):
    # In the final version, this will call HP's ingestion_service.update_limits()
    print(f"Applying new limits: {settings.dict()}")
    return {"status": "ok", "updated": settings.dict()}