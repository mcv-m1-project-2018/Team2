import cv2
from typing import List, Tuple

from numba import njit

from model import Rectangle
from .move_window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER
import numpy as np
from methods.window import combine_overlapped_regions, clear_non_region_mask


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    kernel = np.ones((SIDE, SIDE))

    regions = []
    m = mask
    for _ in range(INTERMEDIATE_STEPS):
        conv_mask = cv2.copyMakeBorder(m, int(SIDE / 2), int(SIDE / 2), int(SIDE / 2), int(SIDE / 2),
                                       cv2.BORDER_CONSTANT, value=0)
        cv2.filter2D(conv_mask, cv2.CV_32F, kernel, delta=1 / SIDE ** 2, borderType=cv2.BORDER_CONSTANT)

        width, height = m.shape
        positions = conv_iter(conv_mask, width, height)
        print(len(positions))
        for pos in positions:
            rec = Rectangle(
                top_left=(pos[0] - int(SIDE / 2), pos[1] - int(SIDE / 2)),
                width=SIDE,
                height=SIDE
            )
            regions.append(rec)

        m = cv2.resize(m, (0, 0), fx=SHRINK_MULTIPLIER, fy=SHRINK_MULTIPLIER)

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)

    return mask, regions


@njit()
def conv_iter(conv_mask: np.array, width: int, height: int) -> List[Tuple[int, int]]:
    ret = []
    for i in range(width):
        for j in range(height):
            if conv_mask[i, j] / SIDE**2 > THRESHOLD:
                ret.append((i, j))

    return ret
