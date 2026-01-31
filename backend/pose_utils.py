import numpy as np

def split_keypoints(flat):
    """
    flat: (T, 384)
    """
    T = flat.shape[0]
    x = flat.reshape(T, 128, 3)

    return {
        "pose":  x[:, :33, :],
        "lhand": x[:, 33:54, :],
        "rhand": x[:, 54:75, :],
        "face":  x[:, 75:, :]
    }
