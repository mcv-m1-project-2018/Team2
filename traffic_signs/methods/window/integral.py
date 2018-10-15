import cv2
from typing import List

from model import Rectangle
from .window import THRESHOLD, SIDE, INTERMEDIATE_STEPS, SHRINK_MULTIPLIER
import numpy as np
from methods.window import combine_overlapped_regions, clear_non_region_mask


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    integral = cv2.integral(mask)

    width, height = mask.shape

    side = SIDE

    regions = []
    for _ in range(INTERMEDIATE_STEPS):
        for i in range((SIDE - 1) / 2, height - (SIDE - 1) / 2):
            for j in range((SIDE - 1) / 2, width - (SIDE - 1) / 2):
                if integral[i, j] / SIDE ** 2 > THRESHOLD:
                    rec = Rectangle(
                        top_left=(i - int(SIDE / 2), j - int(SIDE / 2)),
                        width=SIDE,
                        height=SIDE
                    )
                    regions.append(rec)

        side += int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    regions = combine_overlapped_regions(regions)
    mask = clear_non_region_mask(mask, regions)

    return mask, regions
