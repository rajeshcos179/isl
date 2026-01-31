import os
import sys
import uuid
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

# Pydantic Models
class TextInput(BaseModel):
    text: str

from extract_keypoints import extract_from_video, save_keypoints
from draw_keypoints_video import npz_to_video

from g2p_model import GlossToPose, PoseToGloss
from smoothing import temporal_smooth
from pose_utils import split_keypoints

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

# Load pose to gloss model
try:
    pose_to_gloss_checkpoint = torch.load(POSE2GLOSS_MODEL_PATH, map_location=device)
    print(f"Pose2Gloss checkpoint type: {type(pose_to_gloss_checkpoint)}")
    
    # Extract gloss vocabulary from checkpoint
    gloss2id_pose = pose_to_gloss_checkpoint.get('gloss2id', {})
    id2gloss_pose = {v: k for k, v in gloss2id_pose.items()}
    num_classes = len(gloss2id_pose) if gloss2id_pose else 1000
    
    # Input dim: pose (33*3) + left_hand (21*3) + right_hand (21*3) = 225
    input_dim = 225
    
    pose_to_gloss_model = PoseToGloss(input_dim=input_dim, num_classes=num_classes)
    
    # Load state dict
    if 'model' in pose_to_gloss_checkpoint:
        pose_to_gloss_model.load_state_dict(pose_to_gloss_checkpoint['model'])
    
    pose_to_gloss_model = pose_to_gloss_model.to(device)
    pose_to_gloss_model.eval()
    print(f"Pose2Gloss model loaded successfully with {num_classes} gloss classes")
except Exception as e:
    print(f"Error loading pose2gloss model: {e}")
    import traceback
    traceback.print_exc()
    gloss2id_pose = {}
    id2gloss_pose = {}
    pose_to_gloss_model = None

# Load gloss to pose model
gloss_to_pose_checkpoint = torch.load(GLOSS2POSE_MODEL_PATH, map_location=device)
if isinstance(gloss_to_pose_checkpoint, dict) and 'model' in gloss_to_pose_checkpoint:
    gloss_to_pose_model = gloss_to_pose_checkpoint['model']
else:
    gloss_to_pose_model = gloss_to_pose_checkpoint

print("Models loaded successfully.")

# Load gloss2pose checkpoint for text-to-sign endpoint
ckpt = torch.load("./models/glosstopose.pt", map_location="cpu")
gloss2id = ckpt["gloss2id"]

model = GlossToPose(len(gloss2id))
model.load_state_dict(ckpt["model"])
model.eval()

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


def pose_to_gloss(npz_path):
    """
    Convert keypoints from NPZ file to gloss text using pose2gloss model.
    Returns the predicted gloss text.
    """
    try:
        if pose_to_gloss_model is None:
            return "Model not loaded"
            
        # Load keypoints from NPZ
        data = np.load(npz_path)
        pose = data['pose'][:, :, :3]  # (T, 33, 3) - get xyz coords
        lhand = data['lhand']  # (T, 21, 3)
        rhand = data['rhand']  # (T, 21, 3)
        
        # Concatenate: pose (33*3) + left_hand (21*3) + right_hand (21*3) = 225 features
        x = np.concatenate([pose, lhand, rhand], axis=1)  # (T, 225)
        x = x.reshape(x.shape[0], -1)  # Flatten if needed
        
        # Pad or truncate to MAX_T = 64
        MAX_T = 64
        if x.shape[0] >= MAX_T:
            x = x[:MAX_T]
        else:
            pad = np.zeros((MAX_T - x.shape[0], x.shape[1]))
            x = np.concatenate([x, pad], axis=0)
        
        x = torch.tensor(x).unsqueeze(0).float()  # (1, 64, 225)
        
        with torch.no_grad():
            # Model outputs logits for each gloss class
            logits = pose_to_gloss_model(x.to(device))  # (1, num_classes)
            predicted_gloss_id = logits.argmax(dim=1).item()
        
        # Convert gloss ID to gloss text
        gloss_text = id2gloss_pose.get(predicted_gloss_id, f"gloss_{predicted_gloss_id}")
        
        print(f"Predicted gloss ID: {predicted_gloss_id}, gloss: {gloss_text}")
        
        return gloss_text
    except Exception as e:
        print(f"Error in pose_to_gloss: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"


@app.post("/video-to-text")
async def video_to_text(file: UploadFile = File(...)):
    """
    Convert video to text using pose2gloss model.
    1. Extract keypoints from video
    2. Convert keypoints to gloss using pose2gloss model
    3. Return the gloss text
    """
    uid = str(uuid.uuid4())

    video_path = os.path.join(UPLOAD_DIR, f"{uid}.mp4")
    npz_path = os.path.join(OUTPUT_DIR, f"{uid}.npz")

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 1. Extract keypoints
    frames = extract_from_video(video_path, fps_sample=25)
    save_keypoints(frames, npz_path)

    # 2. Pose → Gloss (using the pose to gloss model)
    gloss = pose_to_gloss(npz_path)

    return {
        "gloss": gloss,
        "text": gloss
    }


@app.post("/text-to-pose")
def text_to_pose(input_data: TextInput):
    """
    Convert text to pose using Gloss2Pose model.
    1. Use text as gloss
    2. Look up gloss ID
    3. Generate pose sequence using GlossToPose model
    4. Smooth the sequence
    5. Split into keypoints (pose, lhand, rhand, face)
    6. Return as JSON
    """
    try:
        text = input_data.text.lower().strip()
        
        # Handle multiple glosses separated by spaces or commas
        glosses = [g.strip() for g in text.replace(',', ' ').split()]
        
        pose_data = {}
        
        for gloss in glosses:
            print(f"Processing gloss: {gloss}")
            
            # Check if gloss exists in vocabulary
            if gloss not in gloss2id:
                print(f"⚠️ Warning: Gloss '{gloss}' not found in vocabulary. Using first gloss as default.")
                # Use the first gloss if not found
                gloss = list(gloss2id.keys())[0]
            
            # Get gloss ID
            gid = torch.tensor([gloss2id[gloss]])
            
            # Generate pose sequence
            with torch.no_grad():
                seq = model(gid)[0].cpu().numpy()  # (64, 384)
            
            # Smooth the sequence
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Gloss2Pose Model"))
            from smoothing import temporal_smooth as g2p_smooth
            from pose_utils import split_keypoints as g2p_split_keypoints
            
            seq_smooth = g2p_smooth(seq, alpha=0.7)
            
            # Split into keypoints
            kp = g2p_split_keypoints(seq_smooth)
            
            # Convert to JSON-serializable format
            pose_data[gloss] = {k: v.tolist() for k, v in kp.items()}
            
            print(f"✅ Generated pose for '{gloss}'")
        
        return {
            "status": "success",
            "pose_data": pose_data,
            "glosses": glosses
        }
        
    except Exception as e:
        print(f"Error in text_to_pose: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

