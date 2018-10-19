import cv2
from typing import List, Tuple

from functional import seq
from numba import njit

from model import Rectangle
from .move_window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER, STEP_FACTOR
import numpy as np
from methods.window import combine_overlapped_regions, clear_non_region_mask


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    integral = cv2.integral(mask)

    width, height = mask.shape

    positions = int_iter(integral, width, height, SIDE)
    regions = []
    for pos in positions:
        regions.append(
            Rectangle(
                top_left=(pos[0] - int(SIDE / 2), pos[1] - int(SIDE / 2)),
                width=SIDE,
                height=SIDE
            )
        )

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)
    return mask, regions


@njit()
def int_iter(integral: np.array, width: int, height: int, initial_side: int) -> List[Tuple[int, int]]:
    ret = []
    side = initial_side
    for _ in range(INTERMEDIATE_STEPS):
        for i in range(int((side - 1) / 2), width - int((side - 1) / 2), int(side * STEP_FACTOR)):
            for j in range(int((side - 1) / 2), height - int((side - 1) / 2), int(side * STEP_FACTOR)):
                if integral[i, j] / initial_side ** 2 > THRESHOLD:
                    ret.append((i, j))
        side = int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    return ret
