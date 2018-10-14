from typing import List

import cv2

from model import Rectangle
import numpy as np

SIDE = 201
INTERMEDIATE_STEPS = 10
STEP_FACTOR = 0.2
SHRINK_MULTIPLIER = .9

THRESHOLD = 0.5


def get_mask(mask: np.array) -> (np.array, List[Rectangle]):
    window = Rectangle(
        top_left=(0, 0),
        width=SIDE + 1,
        height=SIDE + 1
    )
    move_step = SIDE * STEP_FACTOR

    regions = []
    m = mask
    for _ in range(INTERMEDIATE_STEPS):
        width, height = m.shape

        x = 0
        while x + SIDE < width:
            y = 0
            while y + SIDE < height:
                window.top_left = (y, x)

                count = 0
                for i in range(y, y + SIDE + 1):
                    for j in range(x, x + SIDE + 1):
                        if mask[i, j] > 0:
                            count += 1

                if count / window.get_area() > THRESHOLD:
                    regions.append(window.clone())

                y += move_step
            x += move_step

        m = cv2.resize(m, 0, fx=SHRINK_MULTIPLIER, fy=SHRINK_MULTIPLIER)

    return mask, regions
