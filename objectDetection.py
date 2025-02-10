import cv2
import numpy as np
import cv2 as cv

framewidth = 640
frameheight = 480
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("http://192.168.18.139:81/stream")
cap.set(3, framewidth)
cap.set(4, frameheight)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame1 = cap.read()
    frame = cv2.imread("Sans titre.png")
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # frame=cv2.flip(frame,1)
    imgblur = cv2.GaussianBlur(frame, (7, 7), 1)
    hsv = cv.cvtColor(imgblur, cv.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    kernal = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame,
                              mask=red_mask)

    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(frame, frame,
                                mask=green_mask)

    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(frame, frame,
                               mask=blue_mask)

    cx = 1000
    cy = 1000
    contoursgreen, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursred, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursblue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contoursred:
        area = cv2.contourArea(contour)
        if (area > 1000):
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "red", (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            elif len(approx) == 5:
                shape = "pentagon"
            elif len(approx) == 6:
                shape = "hexagon"
            elif len(approx) == 10 or len(approx) == 12:
                shape = "star"
            else:
                shape = "circle"
            cv2.circle(frame, (cx, cy), 1, (0, 0, 0), 2)
            cv2.putText(frame, shape, (cx + 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cx), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cy), (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for contour in contoursgreen:
        area = cv2.contourArea(contour)
        if (area > 300):
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "green", (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            elif len(approx) == 5:
                shape = "pentagon"
            elif len(approx) == 6:
                shape = "hexagon"
            elif len(approx) == 10 or len(approx) == 12:
                shape = "star"
            else:
                shape = "circle"
            cv2.circle(frame, (cx, cy), 1, (0, 0, 0), 2)
            cv2.putText(frame, shape, (cx + 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cx), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cy), (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for contour in contoursblue:
        area = cv2.contourArea(contour)
        if (area > 300):
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 3)
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "blue", (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            elif len(approx) == 5:
                shape = "pentagon"
            elif len(approx) == 6:
                shape = "hexagon"
            elif len(approx) == 10 or len(approx) == 12:
                shape = "star"
            else:
                shape = "circle"
            cv2.circle(frame, (cx, cy), 1, (0, 0, 0), 2)
            cv2.putText(frame, shape, (cx + 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cx), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(cy), (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
