import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
from functional import seq
from numba import njit

from model import Frame

MAX_SIDE = 500
SHOW_OUTPUT = True


def get_frame_with_lines(im: np.array) -> Frame:
    scale = min(MAX_SIDE / im.shape[0], MAX_SIDE / im.shape[1])
    resized = cv2.resize(im, (0, 0), fx=scale, fy=scale)

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

    imres = None
    if SHOW_OUTPUT:
        imres = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    points = []
    angle = 0
    if len(intersections) > 4:

        intersections = simplify_intersections(intersections)

        data = np.array(intersections, dtype=np.float64)

        points, angle = magic(data, im.shape[0] * scale, im.shape[1] * scale)

        if SHOW_OUTPUT:
            for p in points:
                cv2.circle(imres, (int(p[0]), int(p[1])), 3, (255, 0, 0), thickness=-1)

    if SHOW_OUTPUT:
        plt.imshow(imres)
        plt.show()

    # Undo scale
    points = (seq(points)
              .map(lambda point: (int(point[0] / scale), int(point[1] / scale)))
              .to_list())

    return Frame(points, angle)


def line_intersection(line1: ((float, float), (float, float)), line2: ((float, float), (float, float))) -> (
        float, float):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

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
def magic(data: np.array, width: int, height: int):
    """
    For each 3 points with an angle of 90º among them we calculate a 4th one, then, we store
    the set with the biggest area.
    :param data: matrix with rows (x, y, angle)
    :param width: width of the image
    :param height: height of the image
    :return: the 4 points with the biggest area and their angle in interval [0, 90)
    """
    max_area = 0
    points = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]
    angle = 0

    for p0 in range(len(data)):
        for p1 in range(p0 + 1, len(data)):
            for p2 in range(p1 + 1, len(data)):
                if (np.abs(data[p0, 2] - data[p1, 2]) < 5 and
                        np.abs(data[p0, 2] - data[p2, 2]) < 5):

                    d01 = np.subtract(data[p1, :], data[p0, :])
                    d12 = np.subtract(data[p2, :], data[p1, :])
                    d20 = np.subtract(data[p0, :], data[p2, :])

                    # Calculate fouth point
                    if np.linalg.norm(d01) > np.linalg.norm(d12) and np.linalg.norm(d01) > np.linalg.norm(d20):
                        # p2 corner
                        point4 = np.add(data[p1, :], d20)
                    elif np.linalg.norm(d12) > np.linalg.norm(d20):
                        # p0 corner
                        point4 = np.add(data[p2, :], d01)
                    else:
                        # p1 corner
                        point4 = np.add(data[p0, :], d12)

                    # Discard point if outside of image
                    if point4[0] < 0 or point4[1] < 0 or point4[0] > width or point4[1] > height:
                        continue

                    # Combine points in a matrix
                    p = np.vstack((
                        data[p0, 0:2].ravel(),
                        data[p1, 0:2].ravel(),
                        data[p2, 0:2].ravel(),
                        point4[0:2].ravel()
                    ))

                    # Calculate area using shoelace formula
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
                        angle = data[p0, 2]

    return points, angle


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
