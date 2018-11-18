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
SHOW_OUTPUT = False


def get_frame_with_lines(im: np.ndarray) -> Frame:
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

    imres = None
    if SHOW_OUTPUT:
        imres = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    intersections = get_intersections(lines)

    points = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]
    angle = 0
    if len(intersections) > 4:
        points, angle = magic(intersections, im.shape[0] * scale, im.shape[1] * scale)

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


@njit()
def get_intersections(lines: np.ndarray) -> np.array:
    """
    Obtain the lines intersections of around 90º
    :param lines: the set of lines
    :return: a list of points for each intersection and the angle of the lines in the interval [0, 90)
    """
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x, y = line_intersection(
                np.stack((lines[i, 0, 0:2], lines[i, 0, 2:4])),
                np.stack((lines[j, 0, 0:2], lines[j, 0, 2:4]))
            )

            if x != -1 and y != -1:
                anglei = math.degrees(math.atan2(lines[i, 0, 1] - lines[i, 0, 3], lines[i, 0, 0] - lines[i, 0, 2]))
                anglej = math.degrees(math.atan2(lines[j, 0, 1] - lines[j, 0, 3], lines[j, 0, 0] - lines[j, 0, 2]))
                dif = (anglei - anglej) % 180
                if 85 < dif < 95:
                    intersections.append([float(x), float(y), anglei % 90])

    return simplify_intersections(np.array(intersections, dtype=np.float64))


@njit()
def line_intersection(line1: np.ndarray, line2: np.ndarray) -> (float, float):
    xdiff = np.array((line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]))
    ydiff = np.array((line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]))

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = np.array((det(line1[0], line1[1]), det(line2[0], line2[1])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


@njit()
def det(a: np.ndarray, b: np.ndarray):
    return a[0] * b[1] - a[1] * b[0]


@njit()
def magic(data: np.ndarray, width: int, height: int):
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

                    # Combine points in a matrix
                    p = np.vstack((
                        data[p0, 0:2].ravel(),
                        data[p1, 0:2].ravel(),
                        data[p2, 0:2].ravel()
                    ))

                    # Calculate area using shoelace formula
                    area = 0.0
                    for i in range(p.shape[0]):
                        j = (i + 1) % p.shape[0]
                        area += p[i][0] * p[j][1]
                        area -= p[j][0] * p[i][1]
                    area = abs(area) / 2.0

                    if area > max_area:
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

                        max_area = area
                        points = [
                            (data[p0, 0], data[p0, 1]),
                            (data[p1, 0], data[p1, 1]),
                            (data[p2, 0], data[p2, 1]),
                            (point4[0], point4[1])
                        ]
                        angle = data[p0, 2]

    return points, angle


@njit()
def simplify_intersections(data: np.ndarray) -> np.ndarray:
    mask = np.ones((data.shape[0],), np.bool_)
    i = 0
    while i < len(data):
        j = i + 1
        if mask[i]:
            while j < len(data):
                if (mask[j] and
                        math.sqrt(math.pow(data[i, 0] - data[j, 0], 2) + math.pow(data[i, 1] - data[j, 1], 2)) < 5 and
                        np.abs(data[i, 2] - data[j, 2]) < 5):
                    mask[j] = False

                j += 1

        i += 1

    ret = np.zeros((mask.sum(), data.shape[1]), dtype=np.float64)
    ret_pos = 0
    for i in range(data.shape[0]):
        if mask[i]:
            ret[ret_pos] = data[i]
            ret_pos += 1

    return ret
