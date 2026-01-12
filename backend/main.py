import os
import uuid
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extract_keypoints import extract_from_video, save_keypoints
from draw_keypoints_video import npz_to_video

# Paths

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSE2GLOSS_MODEL_PATH = f"{MODEL_DIR}/posetogloss.pt"
GLOSS2POSE_MODEL_PATH = f"{MODEL_DIR}/glosstopose.pt"

# Load Models

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models...")

pose_to_gloss_model = torch.load(POSE2GLOSS_MODEL_PATH, map_location=device)
# pose_to_gloss_model.eval()

gloss_to_pose_model = torch.load(GLOSS2POSE_MODEL_PATH, map_location=device)
# gloss_to_pose_model.eval()

print("Models loaded successfully.")


app = FastAPI(title="Bidirectional ISL Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated videos
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# Routes

@app.get("/")
def root():
    return {"status": "ISL Backend Running"}



@app.post("/video-to-text")
async def video_to_text(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    video_path = os.path.join(UPLOAD_DIR, f"{uid}.mp4")
    npz_path = os.path.join(OUTPUT_DIR, f"{uid}.npz")

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 1. Extract keypoints
    frames = extract_from_video(video_path, fps_sample=25)
    save_keypoints(frames, npz_path)

    # 2. Pose → Gloss
    gloss = pose_to_gloss(npz_path)

    # 3. Gloss → English
    english_text = gloss_to_english(gloss)

    return {
        "gloss": gloss,
        "english_text": english_text
    }



@app.post("/text-to-sign")
async def text_to_sign(data):
    uid = str(uuid.uuid4())

    gloss = english_to_gloss(data.text)

    pose_npz_path = os.path.join(OUTPUT_DIR, f"{uid}_pose.npz")
    video_path = os.path.join(OUTPUT_DIR, f"{uid}.mp4")

    # 1. Gloss → Pose
    gloss_to_pose(gloss, pose_npz_path)

    # 2. Pose → Video
    npz_to_video(
        npz_path=pose_npz_path,
        output_video_path=video_path,
        frame_size=(640, 480),
        fps=25
    )

    return {
        "input_text": data.text,
        "gloss": gloss,
        "animation_url": f"http://localhost:8000/outputs/{uid}.mp4"
    }
