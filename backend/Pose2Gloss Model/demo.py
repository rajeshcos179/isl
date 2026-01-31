# demo.py
import torch
import numpy as np

from model import PoseToGloss

ckpt = torch.load("pose2gloss.pt", map_location="cpu")
gloss2id = ckpt["gloss2id"]
id2gloss = {v: k for k, v in gloss2id.items()}

model = PoseToGloss(225, len(gloss2id))
model.load_state_dict(ckpt["model"])
model.eval()

data = np.load("keypoints/7DFNLpdTyc8_1.npz")

pose = data["pose"][:, :, :3]
lh = data["lhand"]
rh = data["rhand"]

x = np.concatenate([pose, lh, rh], axis=1)
x = x.reshape(x.shape[0], -1)

# pad/truncate
MAX_T = 64
if x.shape[0] >= MAX_T:
    x = x[:MAX_T]
else:
    pad = np.zeros((MAX_T - x.shape[0], x.shape[1]))
    x = np.concatenate([x, pad], axis=0)

x = torch.tensor(x).unsqueeze(0).float()

with torch.no_grad():
    pred = model(x).argmax(dim=1).item()

print("Predicted gloss:", id2gloss[pred])
