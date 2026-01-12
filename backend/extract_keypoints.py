################################################################################
### FILE: extract_keypoints.py
################################################################################
"""
Extract Mediapipe Holistic keypoints from mp4 videos and save per-video npz file.
Outputs: for each uid -> uid.npz containing arrays: 'pose', 'left_hand', 'right_hand', 'face'
"""
import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def extract_from_video(path, max_frames=None, fps_sample=30):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_sample
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    holistic = mp_holistic.Holistic(static_image_mode=False,
                                    model_complexity=1,
                                    enable_segmentation=False,
                                    refine_face_landmarks=True,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    frames = []
    step = max(1, int(round(fps / fps_sample))) if fps_sample>0 else 1
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        # pose: 33 * (x,y,z,visibility)
        pose = np.zeros((33,4), dtype=np.float32)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.z, lm.visibility]
        # left hand: 21 * (x,y,z)
        lhand = np.zeros((21,3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                lhand[i] = [lm.x, lm.y, lm.z]
        rhand = np.zeros((21,3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                rhand[i] = [lm.x, lm.y, lm.z]
        face = np.zeros((468,3), dtype=np.float32)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            max_supported = 468
            count = min(len(face_lms), max_supported)
            for i in range(count):
                lm = face_lms[i]
                face[i] = [lm.x, lm.y, lm.z]
        frames.append({'pose':pose, 'lhand':lhand, 'rhand':rhand, 'face':face})
        idx += 1
        if max_frames and len(frames)>=max_frames:
            break
    cap.release()
    holistic.close()
    return frames


def save_keypoints(frames, outpath):
    # Convert lists of landmarks into arrays: T x ...
    if len(frames)==0:
        np.savez_compressed(outpath)
        return
    pose = np.stack([f['pose'] for f in frames])  # (T,33,4)
    lhand = np.stack([f['lhand'] for f in frames])
    rhand = np.stack([f['rhand'] for f in frames])
    face = np.stack([f['face'] for f in frames])
    np.savez_compressed(outpath, pose=pose, lhand=lhand, rhand=rhand, face=face)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = []
    import csv
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            df.append(r)
    for row in tqdm(df, desc='videos'):
        uid = row['uid']
        video_fn = uid if uid.lower().endswith('.mp4') else uid + '.mp4'
        video_path = os.path.join(args.videos_dir, video_fn)
        if not os.path.exists(video_path):
            tqdm.write(f"Missing video: {video_path}")
            continue
        frames = extract_from_video(video_path, max_frames=args.max_frames, fps_sample=args.fps_sample)
        outpath = os.path.join(args.out_dir, uid + '.npz')
        save_keypoints(frames, outpath)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--videos_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--fps_sample', type=int, default=30)
    args = parser.parse_args()
    main(args)
