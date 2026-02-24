import cv2
import numpy as np
from ultralytics import YOLO

LEFT_URL  = "http://192.168.0.116:81/stream"
RIGHT_URL = "http://192.168.0.117:81/stream"

cap_left = cv2.VideoCapture(LEFT_URL, cv2.CAP_FFMPEG)
cap_right = cv2.VideoCapture(RIGHT_URL, cv2.CAP_FFMPEG)

cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Load model once
model = YOLO("yolov8n.pt")

# Load calibration
params = np.load("stereo_params.npz")
f = float(params["fx"])
B = float(params["baseline_cm"])

print("Model loaded. Classes:", model.names)

def compute_xyz(xL, yL, d, img_w, img_h):
    cx, cy = img_w / 2, img_h / 2
    Z = (f * B) / max(d, 1.0)
    X = (xL - cx) * Z / f
    Y = (yL - cy) * Z / f
    return X, Y, Z

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not (retL and retR):
        continue

    h, w, _ = frameL.shape

    # Run detection on left frame (faster)
    resultsL = model(frameL, conf=0.4, verbose=False)[0]
    resultsR = model(frameR, conf=0.4, verbose=False)[0]

    boxesL = resultsL.boxes
    boxesR = resultsR.boxes

    # Simple matching: same class, nearest x
    for bL in boxesL:
        x1, y1, x2, y2 = bL.xyxy[0].cpu().numpy()
        clsL = int(bL.cls[0].item())
        confL = float(bL.conf[0].item())
        xL = (x1 + x2) / 2
        yL = (y1 + y2) / 2

        # find best match in right frame
        best = None
        best_dx = 1e9
        for bR in boxesR:
            clsR = int(bR.cls[0].item())
            if clsR != clsL:
                continue
            xr1, yr1, xr2, yr2 = bR.xyxy[0].cpu().numpy()
            xR = (xr1 + xr2) / 2
            dx = abs(xL - xR)
            if dx < best_dx:
                best_dx = dx
                best = (xr1, yr1, xr2, yr2, xR)

        if best is None:
            continue

        _, _, _, _, xR = best
        d = abs(xL - xR)

        X, Y, Z = compute_xyz(xL, yL, d, w, h)

        label = model.names[clsL]
        text = f"{label}  Z:{Z:.1f}cm  X:{X:.1f} Y:{Y:.1f}"

        # Draw bbox + text on left frame
        cv2.rectangle(frameL, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(frameL, text, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("Left (3D Overlay)", frameL)
    cv2.imshow("Right", frameR)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()