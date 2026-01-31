import torch
import json

from g2p_model import GlossToPose
from smoothing import temporal_smooth
from pose_utils import split_keypoints

# -----------------------------
# Load model
# -----------------------------
ckpt = torch.load("gloss2pose.pt", map_location="cpu")
gloss2id = ckpt["gloss2id"]

model = GlossToPose(len(gloss2id))
model.load_state_dict(ckpt["model"])
model.eval()

# -----------------------------
# Choose gloss
# -----------------------------
gloss = "air"
gid = torch.tensor([gloss2id[gloss]])

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    seq = model(gid)[0].cpu().numpy()   # (64, 384)

# -----------------------------
# Smooth
# -----------------------------
seq = temporal_smooth(seq, alpha=0.7)

# -----------------------------
# Split
# -----------------------------
kp = split_keypoints(seq)

# -----------------------------
# Save JSON
# -----------------------------
out = {k: v.tolist() for k, v in kp.items()}

with open(f"{gloss}.json", "w") as f:
    json.dump(out, f)

print("✅ Saved:", f"{gloss}.json")
