import cv2
import numpy as np
import matplotlib.pyplot as plt

params = np.load("stereo_params.npz")
f = float(params["fx"])           # focal length in pixels
B = float(params["baseline_cm"])  # baseline in cm

print("Using f =", f, "px, B =", B, "cm")

left_path  = "calib/left/img_003.jpg"
right_path = "calib/right/img_003.jpg"

imgL = cv2.imread(left_path)
imgR = cv2.imread(right_path)

h, w = imgL.shape[:2]
cx, cy = w / 2.0, h / 2.0   # principal point approx (better if from calibration)

pts = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        pts.append((event.xdata, event.ydata))
        print("Clicked:", pts[-1])
        if len(pts) == 2:
            plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)); ax[0].set_title("Left – click point")
ax[1].imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)); ax[1].set_title("Right – click same point")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

(xL, yL), (xR, yR) = pts

d = abs(xL - xR)
print("Disparity (px):", d)

Z = (f * B) / d
print("Estimated depth Z (cm):", Z)

# ---- 3D coordinates ----
X_left = (xL - cx) * Z / f
Y_left = (yL - cy) * Z / f

# shift origin to midpoint of baseline
X_mid = X_left - (B / 2.0)
Y_mid = Y_left
Z_mid = Z

print("\n3D coordinates (cm) in mid-baseline frame:")
print(f"X = {X_mid:.2f} cm")
print(f"Y = {Y_mid:.2f} cm")
print(f"Z = {Z_mid:.2f} cm")