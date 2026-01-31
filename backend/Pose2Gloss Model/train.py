# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from dataset import PoseGlossDataset
from model import PoseToGloss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 35
LR = 1e-4

# ---------- Load filtered train CSV ----------
train_df = pd.read_csv("train_filt.csv")

# ---------- Build vocab ----------
glosses = sorted(train_df["gloss"].unique())
gloss2id = {g: i for i, g in enumerate(glosses)}
num_classes = len(gloss2id)

# ---------- Dataset ----------
train_ds = PoseGlossDataset(
    csv_file="train_filt.csv",
    kp_dir="keypoints/",
    gloss2id=gloss2id
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------- Model ----------
model = PoseToGloss(input_dim=225, num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- Train ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

torch.save({
    "model": model.state_dict(),
    "gloss2id": gloss2id
}, "pose2gloss.pt")

print("✅ Training complete")
