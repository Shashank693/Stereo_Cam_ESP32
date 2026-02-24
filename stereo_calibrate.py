import cv2
import glob
import numpy as np

CHECKERBOARD = (8, 5)   # inner corners
SQUARE_SIZE = 3      # cm (measure your print!)

left_images = sorted(glob.glob("calib/left/*.jpg"))
right_images = sorted(glob.glob("calib/right/*.jpg"))

assert len(left_images) == len(right_images)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

img_size = None

for l_path, r_path in zip(left_images, right_images):
    img_l = cv2.imread(l_path)
    img_r = cv2.imread(r_path)

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)

    if ret_l and ret_r:
        objpoints.append(objp)

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        img_size = gray_l.shape[::-1]

ret_l, K1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
ret_r, K2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)

ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K1, dist1, K2, dist2, img_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)
print("Calibration image size:", img_size)
baseline_cm = np.linalg.norm(T)
fx = K1[0,0]

np.savez("stereo_params.npz",
         K1=K1, dist1=dist1,
         K2=K2, dist2=dist2,
         R=R, T=T,
         baseline_cm=baseline_cm,
         fx=fx)

print("Saved stereo_params.npz")
print("Baseline (cm):", baseline_cm)
print("Focal length fx (px):", fx)