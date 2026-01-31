import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from g2p_model import GlossToPose
from g2p_dataset import GlossPoseDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_df = pd.read_csv("train_filt.csv")
glosses = sorted(train_df["gloss"].unique())
gloss2id = {g:i for i,g in enumerate(glosses)}

dataset = GlossPoseDataset("train_filt_f4.csv", "keypoints/", gloss2id)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = GlossToPose(len(gloss2id)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(30):
    model.train()
    total_loss = 0

    for gloss_id, target_pose in loader:
        gloss_id = gloss_id.to(DEVICE)
        target_pose = target_pose.to(DEVICE)

        pred = model(gloss_id)
        loss = criterion(pred, target_pose)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}  Loss: {total_loss/len(loader):.4f}")

torch.save({
    "model": model.state_dict(),
    "gloss2id": gloss2id
}, "gloss2pose.pt")
