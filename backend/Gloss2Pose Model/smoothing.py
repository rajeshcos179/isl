import numpy as np

def temporal_smooth(sequence, alpha=0.7):
    """
    sequence: (T, D)
    alpha: smoothing factor (0.6-0.8 works well)
    """
    smoothed = np.zeros_like(sequence)
    smoothed[0] = sequence[0]

    for t in range(1, sequence.shape[0]):
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * sequence[t]

    return smoothed
