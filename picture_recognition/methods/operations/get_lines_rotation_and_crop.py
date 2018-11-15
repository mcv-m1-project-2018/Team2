import itertools
import math
from typing import List

import numba

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def get_lines_rotation_and_crop(im: np.array):
    resized = cv2.resize(im, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # uses the above two partial derivatives
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_gradientx = cv2.convertScaleAbs(sobelx)
    abs_gradienty = cv2.convertScaleAbs(sobely)
    # combine the two in equal proportions
    gray = cv2.addWeighted(abs_gradientx, 0.5, abs_gradienty, 0.5, 0)

    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    gray = cv2.Canny(gray, threshold1=0, threshold2=50, apertureSize=3)

    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    # lines = cv2.HoughLines(gray, 1, np.pi / 180, 150)
    imres = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x, y = line_intersection(
                ((lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3])),
                ((lines[j][0][0], lines[j][0][1]), (lines[j][0][2], lines[j][0][3]))
            )
            if x != -1 and y != -1:
                anglei = math.degrees(math.atan2(lines[i][0][1] - lines[i][0][3], lines[i][0][0] - lines[i][0][2]))
                anglej = math.degrees(math.atan2(lines[j][0][1] - lines[j][0][3], lines[j][0][0] - lines[j][0][2]))
                dif = (anglei - anglej) % 180
                if 85 < dif < 95:
                    intersections.append([float(x), float(y), anglei % 90])

    if len(intersections) > 4:

        intersections = simplify_intersections(intersections)

        data = np.array(intersections, dtype=np.float64)

        points = magic(data)

        for p in points:
            cv2.circle(imres, (int(p[0]), int(p[1])), 3, (255, 0, 0), thickness=-1)

    """for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(i, (x1, y1), (x2, y2), (0, 255, 0), 2)"""

    """for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(i, (x1, y1), (x2, y2), (0, 0, 255), 2)"""

    plt.imshow(imres)
    plt.show()
    print(1)


def line_intersection(line1: ((float, float), (float, float)), line2: ((float, float), (float, float))) -> (
        float, float):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


@njit()
def magic(data):
    max_area = 0
    points = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]

    for p0 in range(len(data)):
        for p1 in range(p0 + 1, len(data)):
            for p2 in range(p1 + 1, len(data)):
                if (np.abs(data[p0, 2] - data[p1, 2]) < 5 and
                        np.abs(data[p0, 2] - data[p2, 2]) < 5):

                    d01 = np.subtract(data[p1, :], data[p0, :])
                    d12 = np.subtract(data[p2, :], data[p1, :])
                    d20 = np.subtract(data[p0, :], data[p2, :])

                    if np.linalg.norm(d01) > np.linalg.norm(d12) and np.linalg.norm(d01) > np.linalg.norm(d20):
                        # p2 corner
                        point4 = np.add(data[p1, :], d20)
                    elif np.linalg.norm(d12) > np.linalg.norm(d20):
                        # p0 corner
                        point4 = np.add(data[p2, :], d01)
                    else:
                        # p1 corner
                        point4 = np.add(data[p0, :], d12)

                    p = np.vstack((
                        data[p0, 0:2].ravel(),
                        data[p1, 0:2].ravel(),
                        data[p2, 0:2].ravel(),
                        point4[0:2].ravel()
                    ))

                    area = 0.0
                    for i in range(4):
                        j = (i + 1) % 4
                        area += p[i][0] * p[j][1]
                        area -= p[j][0] * p[i][1]
                    area = abs(area) / 2.0

                    if area > max_area:
                        max_area = area
                        points = [
                            (data[p0, 0], data[p0, 1]),
                            (data[p1, 0], data[p1, 1]),
                            (data[p2, 0], data[p2, 1]),
                            (point4[0], point4[1])
                        ]

    return points


def simplify_intersections(data: List[List[float]]) -> List[List[float]]:
    i = 0
    while i < len(data):
        j = i + 1
        while j < len(data):
            if (math.sqrt(math.pow(data[i][0] - data[j][0], 2) + math.pow(data[i][1] - data[j][1], 2)) < 5 and
                    np.abs(data[i][2] - data[j][2]) < 5):
                del data[j]
                j -= 1

            j += 1

        i += 1

    return data
