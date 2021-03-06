import imutils as imutils
import cv2
import numpy as np
from model import Rectangle
import matplotlib.pyplot as plt


def detect_text(img: np.ndarray) -> (np.ndarray, Rectangle):
    im = img.copy()
    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    Y, U, V = cv2.split(im_yuv)

    kernel = np.ones((3, 3))

    Ib = cv2.GaussianBlur(Y, (3, 3), 5)

    gradient = cv2.morphologyEx(Ib, cv2.MORPH_GRADIENT, kernel)

    re, th1 = cv2.threshold(gradient, 126, 255, cv2.THRESH_BINARY)

    edges = cv2.dilate(th1, kernel)

    edges1 = cv2.erode(edges, kernel)

    cnts = cv2.findContours(edges1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # ret, labels = cv2.connectedComponents(cnts)

    text = []
    corner_left = Y.shape
    corner_right = (0, 0)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (5 <= w <= 300) and (6 <= h <= 300):

            text.append(c)
            if cv2.norm((x, y)) < cv2.norm(corner_left):
                corner_left = (x, y)
            if cv2.norm((x + w, y + h)) > cv2.norm(corner_right):
                corner_right = (x + w, y + h)

        # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    padding = int(0.07 * (cv2.norm(np.subtract(corner_right, corner_left))))
    sub = np.subtract(corner_right, corner_left)
    width = sub[0]
    height = sub[1]

    bounding = Rectangle(corner_left, width, height)

    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

    # cv2.rectangle(im, (corner_left[0] - padding, corner_left[1] - padding),
    #              (bounding.get_bottom_right()[0] + padding, bounding.get_bottom_right()[1] + padding), (0, 0, 0), -1)

    corner_left = (corner_left[0] - padding, corner_left[1] - padding)
    corner_right = (bounding.get_bottom_right()[0] + padding, bounding.get_bottom_right()[1] + padding)
    mask = cv2.rectangle(mask, corner_left, corner_right, (0, 0, 0), -1)
    sub = np.subtract(corner_right, corner_left)
    width = sub[0]
    height = sub[1]
    bounding = Rectangle(corner_left, width, height)
    # cv2.imshow('mask',mask)

    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()

    return mask, bounding


def detect_text_gray(img: np.ndarray) -> (np.ndarray, Rectangle):
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
    # ret, labels = cv2.connectedComponents(cnts)

    text = []
    corner_left = gray.shape
    corner_right = (0, 0)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (5 <= w <= 150) and (5 <= h <= 150):

            text.append(c)
            if cv2.norm((x, y)) < cv2.norm(corner_left):
                corner_left = (x, y)
            if cv2.norm((x + w, y + h)) > cv2.norm(corner_right):
                corner_right = (x + w, y + h)

        # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    padding = int(0.07 * (cv2.norm(np.subtract(corner_right, corner_left))))
    sub = np.subtract(corner_right, corner_left)
    width = sub[0]
    height = sub[1]

    bounding = Rectangle(corner_left, width, height)

    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

    # cv2.rectangle(im, (corner_left[0] - padding, corner_left[1] - padding),
    #              (bounding.get_bottom_right()[0] + padding, bounding.get_bottom_right()[1] + padding), (0, 0, 0), -1)

    corner_left = (corner_left[0] - padding, corner_left[1] - padding)
    corner_right = (bounding.get_bottom_right()[0] + padding, bounding.get_bottom_right()[1] + padding)
    mask = cv2.rectangle(mask, corner_left, corner_right, (0, 0, 0), -1)

    sub = np.subtract(corner_right, corner_left)
    width = sub[0]
    height = sub[1]
    bounding = Rectangle(corner_left, width, height)
    # cv2.imshow('mask',mask)

    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()

    return mask, bounding
