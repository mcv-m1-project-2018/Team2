import imutils as imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_text(img: np.array) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3))

    Ib = cv2.GaussianBlur(gray, (3, 3), 5)

    gradient = cv2.morphologyEx(Ib, cv2.MORPH_GRADIENT, kernel)

    re, th1 = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    th3 = cv2.bitwise_and(th1, th2)

    edges = cv2.dilate(th3, kernel)

    edges1 = cv2.erode(edges, kernel)

    cnts = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    text = []
    corner_left = gray.shape
    corner_right = (0, 0)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (5 <= w <= 150) and (5 <= h <= 150):

            text.append(c)
            if cv2.norm((x, y)) < cv2.norm(corner_left):
                corner_left = (x, y)
            if cv2.norm((x+w, y+h)) > cv2.norm(corner_right):
                corner_right = (x+w, y+h)

        # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    padding = 0.05 * (cv2.norm(np.subtract(corner_right, corner_left)))

    cv2.rectangle(img, (int(corner_left[0] - padding), int(corner_left[1] - padding)),
                  (int(corner_right[0] + padding), int(corner_right[1] + padding)), (0, 0, 0), -1)

    return img
