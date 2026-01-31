import cv2
import numpy as np
from skeleton import POSE_EDGES, HAND_EDGES

def draw_points(img, pts, color, r=2):
    for x,y,z in pts:
        cv2.circle(img, (int(x), int(y)), r, color, -1)

def draw_edges(img, pts, edges, color):
    for i,j in edges:
        x1,y1,_ = pts[i]
        x2,y2,_ = pts[j]
        cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)

def render_video(kp, out_path, size=512, fps=25):
    T = kp["pose"].shape[0]

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size, size)
    )

    for t in range(T):
        img = np.ones((size,size,3), dtype=np.uint8) * 255

        for part in kp:
            kp[part][t,:,0] *= size
            kp[part][t,:,1] *= size

        draw_edges(img, kp["pose"][t], POSE_EDGES, (0,0,255))
        draw_edges(img, kp["lhand"][t], HAND_EDGES, (0,255,0))
        draw_edges(img, kp["rhand"][t], HAND_EDGES, (255,0,0))

        draw_points(img, kp["pose"][t], (0,0,255), 3)
        draw_points(img, kp["lhand"][t], (0,255,0))
        draw_points(img, kp["rhand"][t], (255,0,0))
        draw_points(img, kp["face"][t], (0,0,0), 1)

        writer.write(img)

    writer.release()
