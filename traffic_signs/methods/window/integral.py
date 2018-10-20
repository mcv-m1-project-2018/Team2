import cv2
from typing import List, Tuple

from functional import seq
from numba import njit

from model import Rectangle
from .move_window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER, STEP_FACTOR
import numpy as np
from methods.window import combine_overlapped_regions, clear_non_region_mask


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    integral = cv2.integral(mask / 255)

    positions = int_iter(integral)
    regions = []
    for pos in positions:
        regions.append(
            Rectangle(
                top_left=(pos[0], pos[1]),
                width=pos[2],
                height=pos[2]
            )
        )

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)
    return mask, regions


@njit()
def int_iter(integral: np.array) -> List[Tuple[int, int, int]]:
    ret = []
    side = SIDE
    for _ in range(INTERMEDIATE_STEPS):
        for i in range(0, integral.shape[0] - side, int(side * STEP_FACTOR)):
            for j in range(0, integral.shape[1] - side, int(side * STEP_FACTOR)):
                s = integral[i + side, j + side] - integral[i + side, j] - integral[i, j + side] + integral[i, j]
                if s / side ** 2 > THRESHOLD:
                    ret.append((i, j, side))

        side = int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    return ret
