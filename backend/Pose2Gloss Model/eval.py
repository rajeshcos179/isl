# eval.py
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import PoseGlossDataset
from model import PoseToGloss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("pose2gloss.pt", map_location=DEVICE)
gloss2id = ckpt["gloss2id"]
id2gloss = {v: k for k, v in gloss2id.items()}

test_ds = PoseGlossDataset(
    csv_file="test_filt.csv",
    kp_dir="keypoints/",
    gloss2id=gloss2id
)

test_loader = DataLoader(test_ds, batch_size=8)

model = PoseToGloss(225, len(gloss2id)).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print("Correct:", correct, "Total:", total)
print("Test Accuracy:", correct / total)
