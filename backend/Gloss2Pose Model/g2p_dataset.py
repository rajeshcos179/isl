import os
import numpy as np
import torch
from torch.utils.data import Dataset
from face_indices import FACE_IDXS

MAX_T = 64

class GlossPoseDataset(Dataset):
    def __init__(self, csv_file, kp_dir, gloss2id):
        import pandas as pd
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

        pose = data["pose"][:, :, :3]
        lh = data["lhand"]
        rh = data["rhand"]
        face = data["face"][:, FACE_IDXS, :]   # (T, F, 3)

        x = np.concatenate([pose, lh, rh, face], axis=1)
        x = x.reshape(x.shape[0], -1)

        if x.shape[0] >= MAX_T:
            x = x[:MAX_T]
        else:
            pad = np.zeros((MAX_T - x.shape[0], x.shape[1]))
            x = np.concatenate([x, pad], axis=0)

        return (
            torch.tensor(self.gloss2id[gloss]),
            torch.tensor(x, dtype=torch.float32)
        )
