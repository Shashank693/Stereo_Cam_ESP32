import cv2
import os
import time

LEFT_URL  = "http://192.168.0.116:81/stream"
RIGHT_URL = "http://192.168.0.117:81/stream"

SAVE_DIR_L = "calib/left"
SAVE_DIR_R = "calib/right"

os.makedirs(SAVE_DIR_L, exist_ok=True)
os.makedirs(SAVE_DIR_R, exist_ok=True)

capL = cv2.VideoCapture(LEFT_URL, cv2.CAP_FFMPEG)
capR = cv2.VideoCapture(RIGHT_URL, cv2.CAP_FFMPEG)

capL.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capR.set(cv2.CAP_PROP_BUFFERSIZE, 1)

idx = 0

print("Press 'c' to capture a stereo pair, 'q' to quit.")

def draw_simple_label(frame, text):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)  # small black strip
    cv2.putText(frame, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        capL.release()
        capR.release()
        time.sleep(0.5)
        capL = cv2.VideoCapture(LEFT_URL, cv2.CAP_FFMPEG)
        capR = cv2.VideoCapture(RIGHT_URL, cv2.CAP_FFMPEG)
        continue

    frameL_disp = draw_simple_label(frameL.copy(), "LEFT CAMERA")
    frameR_disp = draw_simple_label(frameR.copy(), "RIGHT CAMERA")

    view = cv2.hconcat([
        cv2.resize(frameL_disp, (400, 300)),
        cv2.resize(frameR_disp, (400, 300))
    ])

    cv2.imshow("Stereo Capture", view)

    # If user clicks the X button on the window, exit loop
    if cv2.getWindowProperty("Stereo Capture", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        fnameL = os.path.join(SAVE_DIR_L, f"img_{idx:03d}.jpg")
        fnameR = os.path.join(SAVE_DIR_R, f"img_{idx:03d}.jpg")
        cv2.imwrite(fnameL, frameL)
        cv2.imwrite(fnameR, frameR)
        print(f"Saved pair {idx}")
        idx += 1
        time.sleep(0.2)

    elif key == ord('q') or key == 27:
        break


capL.release()
capR.release()
cv2.destroyAllWindows()