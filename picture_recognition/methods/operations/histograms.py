import numpy as np
from enum import Enum
import cv2
from cv2 import COLOR_BGR2Luv


class HistogramTypes(Enum):
    HSV = 0,
    LUV = 1


def get_histogram(im: np.array, histogram_type=HistogramTypes.HSV) -> np.array:
    hist = None
    if histogram_type == HistogramTypes.HSV:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([im], [0, 1, 2], None, [256], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
    elif histogram_type == HistogramTypes.LUV:
        im = cv2.cvtColor(im, cv2, COLOR_BGR2Luv)
        hist = cv2.calcHist([im], [0, 1, 2], None, [256], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

    return hist
