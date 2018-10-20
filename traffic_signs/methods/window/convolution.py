from typing import List, Tuple

import cv2
import numpy as np
from numba import njit

from methods.window import combine_overlapped_regions, clear_non_region_mask
from model import Rectangle
from .move_window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER, STEP_FACTOR


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    regions = []
    side = SIDE
    for _ in range(INTERMEDIATE_STEPS):
        kernel = np.empty((side, side))
        kernel.fill(1 / 255)

        conv_res = cv2.filter2D(mask, cv2.CV_32F, kernel, anchor=(0, 0),
                                borderType=cv2.BORDER_CONSTANT)

        positions = conv_iter(conv_res, side)
        for pos in positions:
            rec = Rectangle(
                top_left=(pos[0], pos[1]),
                width=side,
                height=side
            )
            regions.append(rec)

        side = int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)

    return mask, regions


@njit()
def conv_iter(conv_res: np.array, side: int) -> List[Tuple[int, int]]:
    move_step = int(side * STEP_FACTOR)
    ret = []
    for i in range(int(side / 2), conv_res.shape[0] - int(side / 2), move_step):
        for j in range(int(side / 2), conv_res.shape[1] - int(side / 2), move_step):
            if conv_res[i, j] / side ** 2 > THRESHOLD:
                ret.append((i, j))
    return ret
