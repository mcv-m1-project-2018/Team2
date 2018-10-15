import numpy as np

from methods.operations import segregation
from methods.window import integral


class hsv_integral:

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask, regions = integral(mask)

        return regions, mask, im
