import cv2
from typing import List

from model import Rectangle
from .window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER
import numpy as np


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    kernel = np.ones((SIDE, SIDE))

    regions = []
    m = mask
    for _ in range(INTERMEDIATE_STEPS):
        conv_mask = cv2.copyMakeBorder(m, SIDE / 2, SIDE / 2, SIDE / 2, SIDE / 2, cv2.BORDER_CONSTANT, value=0)
        cv2.filter2D(conv_mask, cv2.CV_32F, kernel, delta=1 / SIDE ** 2, borderType=cv2.BORDER_CONSTANT)

        width, height = m.shape
        for i in range(height):
            for j in range(width):
                if conv_mask[i, j] > THRESHOLD:
                    rec = Rectangle(
                        top_left=(i - int(SIDE / 2), j - int(SIDE / 2)),
                        width=SIDE,
                        height=SIDE
                    )
                    regions.append(rec)

        m = cv2.resize(m, 0, fx=SHRINK_MULTIPLIER, fy=SHRINK_MULTIPLIER)

    return mask, regions
