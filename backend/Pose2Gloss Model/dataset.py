# dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_T = 64  # IMPORTANT

class PoseGlossDataset(Dataset):
    def __init__(self, csv_file, kp_dir, gloss2id):
        self.df = pd.read_csv(csv_file)
        self.kp_dir = kp_dir
        self.gloss2id = gloss2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["uid"]
        gloss = row["gloss"]

        data = np.load(os.path.join(self.kp_dir, uid + ".npz"))

        pose = data["pose"][:, :, :3]     # (T,33,3)
        lh = data["lhand"]                # (T,21,3)
        rh = data["rhand"]                # (T,21,3)

        x = np.concatenate([pose, lh, rh], axis=1)  # (T,75,3)
        x = x.reshape(x.shape[0], -1)               # (T,225)

        # Pad / truncate
        T = x.shape[0]
        if T >= MAX_T:
            x = x[:MAX_T]
        else:
            pad = np.zeros((MAX_T - T, x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        return torch.tensor(x, dtype=torch.float32), self.gloss2id[gloss]
