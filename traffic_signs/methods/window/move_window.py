from typing import List, Tuple

import cv2
import numpy as np
from numba import njit

from model import Rectangle
from . import combine_overlapped_regions, clear_non_region_mask

SIDE = 121
INTERMEDIATE_STEPS = 10
STEP_FACTOR = 0.1
SHRINK_MULTIPLIER = .9
THRESHOLD = 0.25


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    regions = []
    m = mask
    for _ in range(INTERMEDIATE_STEPS):
        width, height = m.shape

        positions = window_iter(mask, width, height)
        for pos in positions:
            regions.append(
                Rectangle(
                    top_left=(pos[0] - int(SIDE / 2), pos[1] - int(SIDE / 2)),
                    width=SIDE,
                    height=SIDE
                )
            )

        m = cv2.resize(m, (0, 0), fx=SHRINK_MULTIPLIER, fy=SHRINK_MULTIPLIER)

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)
    return mask, regions


@njit()
def window_iter(mask: np.array, width: int, height: int) -> List[Tuple[int, int]]:
    move_step = int(SIDE * STEP_FACTOR)
    ret = []
    x = 0
    while x + SIDE < width:
        y = 0
        while y + SIDE < height:
            count = 0
            for i in range(x, x + SIDE + 1):
                for j in range(y, y + SIDE + 1):
                    if mask[i, j] > 0:
                        count += 1

            if count / (SIDE**2) > THRESHOLD:
                ret.append((x, y))

            y += move_step
        x += move_step
    return ret
