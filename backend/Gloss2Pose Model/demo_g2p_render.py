import torch
import numpy as np

from g2p_model import GlossToPose
from smoothing import temporal_smooth
from pose_utils import split_keypoints
from render import render_video

# Load model
ckpt = torch.load("gloss2pose.pt", map_location="cpu")
gloss2id = ckpt["gloss2id"]

model = GlossToPose(len(gloss2id))
model.load_state_dict(ckpt["model"])
model.eval()

# Choose gloss
gloss = "air"
gid = torch.tensor([gloss2id[gloss]])

with torch.no_grad():
    seq = model(gid)[0].numpy()  # (T,225)

# Smooth
seq = temporal_smooth(seq, alpha=0.7)

# Split
kp = split_keypoints(seq)

# Render
render_video(kp, f"{gloss}.mp4")

print("Saved video:", f"{gloss}.mp4")
