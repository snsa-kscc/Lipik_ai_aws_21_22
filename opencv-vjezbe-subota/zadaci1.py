import cv2
import numpy as np
import matplotlib.pyplot as plt

capture = cv2.VideoCapture("car_meanshift.mp4")

if not capture.isOpened():
    print("Error opening video stream or file")
    exit(1)

backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = capture.read()
    if frame is None:
        print("No frame")
        break
    mask = backSub.apply(frame)

    kernel = np.ones((5, 5), np.uint8)
    bez_suma = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Car mean", frame)
    cv2.imshow("Car mean shift", mask)
    cv2.imshow("Car mean shift filtered", bez_suma)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
