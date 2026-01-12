################################################################################
### FILE: draw_keypoints_video.py
################################################################################
"""
Load Mediapipe Holistic keypoints from .npz files, draw them, and save as a video.
"""
import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Drawing settings
DRAW_CONNECTIONS = True

def draw_frame_from_keypoints(frame, pose, lhand, rhand, face):
    """Draw pose, hands, and face landmarks on a frame"""
    # Helper to convert normalized coordinates to pixel coordinates
    h, w = frame.shape[:2]
    def to_px(coords):
        return int(coords[0]*w), int(coords[1]*h)
    
    # Pose landmarks
    if pose is not None:
        for i, lm in enumerate(pose):
            x, y, z, vis = lm
            if vis>0:
                cv2.circle(frame, to_px([x, y]), 3, (0,255,0), -1)
    
    # Left hand
    if lhand is not None:
        for i, lm in enumerate(lhand):
            x, y, z = lm
            cv2.circle(frame, to_px([x, y]), 3, (0,0,255), -1)
    
    # Right hand
    if rhand is not None:
        for i, lm in enumerate(rhand):
            x, y, z = lm
            cv2.circle(frame, to_px([x, y]), 3, (255,0,0), -1)
    
    # Face landmarks
    if face is not None:
        for i, lm in enumerate(face):
            x, y, z = lm
            cv2.circle(frame, to_px([x, y]), 1, (255,255,0), -1)

    return frame

def npz_to_video(npz_path, output_video_path, frame_size=(640,480), fps=30):
    data = np.load(npz_path)
    T = len(data['pose'])

    # Video writer
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )

    for t in tqdm(range(T), desc='Rendering frames'):
        # Create blank frame (black)
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        pose = data['pose'][t] if 'pose' in data else None
        lhand = data['lhand'][t] if 'lhand' in data else None
        rhand = data['rhand'][t] if 'rhand' in data else None
        face = data['face'][t] if 'face' in data else None

        frame = draw_frame_from_keypoints(frame, pose, lhand, rhand, face)
        out.write(frame)

    out.release()
    print(f"Saved video: {output_video_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='Input npz file with keypoints')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    npz_to_video(args.npz, args.output, frame_size=(args.width,args.height), fps=args.fps) 
