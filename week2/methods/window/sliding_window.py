from typing import List, Tuple

import numpy as np
from numba import njit

from model import Rectangle
from . import combine_overlapped_regions, clear_non_region_mask

SIDE = 51
INTERMEDIATE_STEPS = 15
STEP_FACTOR = 0.1
SHRINK_MULTIPLIER = .9
THRESHOLD = 0.6


def sliding_window(mask: np.array) -> (np.array, List[Rectangle]):
    regions = []

    side = SIDE
    for _ in range(INTERMEDIATE_STEPS):
        positions = window_iter(mask, side)
        for pos in positions:
            regions.append(
                Rectangle(
                    top_left=(pos[0], pos[1]),
                    width=side,
                    height=side
                )
            )

        side = int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)
    return mask, regions


@njit()
def window_iter(mask: np.array, side: int) -> List[Tuple[int, int]]:
    move_step = int(side * STEP_FACTOR)
    ret = []
    x = 0
    while x + side < mask.shape[0]:
        y = 0
        while y + side < mask.shape[1]:
            count = 0
            for i in range(x, x + side + 1):
                for j in range(y, y + side + 1):
                    if mask[i, j] > 0:
                        count += 1

            if count / (side ** 2) > THRESHOLD:
                ret.append((x, y))

            y += move_step
        x += move_step
    return ret
