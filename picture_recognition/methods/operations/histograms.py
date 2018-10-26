from enum import Enum
from typing import List

import cv2
import numpy as np


class HistogramTypes(Enum):
    HSV = 0,
    YCbCr = 1


def get_histogram(im: np.array, histogram_type=HistogramTypes.HSV) -> List[np.array]:
    hist = []
    if histogram_type == HistogramTypes.HSV:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        for i in range(im.shape[2]):
            h = cv2.calcHist([im[:, :, i]], [0], None, [256], [0, 256])
            cv2.normalize(h, h)
            hist.append(h)
    elif histogram_type == HistogramTypes.YCbCr:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        for i in range(im.shape[2]):
            h = cv2.calcHist([im[:, :, i]], [0], None, [256], [0, 256])
            cv2.normalize(h, h)
            hist.append(h)

    return hist
