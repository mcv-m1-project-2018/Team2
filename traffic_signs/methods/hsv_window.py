import numpy as np

from methods.operations import segregation
from methods.window import window


class hsv_window:

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask, regions = window(mask)

        return regions, mask, im
