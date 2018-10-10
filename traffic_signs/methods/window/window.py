from model import Rectangle
import numpy as np

MAX_SIZE = 200
MIN_SIZE = 25
INTERMEDIATE_STEPS = 10
STEP_FACTOR = 0.2


class Window:

    def __init__(self):
        pass

    def get_mask(self, mask: np.array):
        size = MAX_SIZE
        step = (MAX_SIZE - MIN_SIZE) / INTERMEDIATE_STEPS

        width, height = mask.shape

        while size >= MIN_SIZE:
            window = Rectangle(
                top_left=(0, 0),
                width=size,
                height=size
            )

            move_step = size * STEP_FACTOR
            x = 0
            while x + size < width:
                y = 0
                while y + size < height:
                    window.top_left = (y, x)
                    # Run code for the area inside the rectangle
                    y += move_step
                x += move_step

            size -= step

        return mask
