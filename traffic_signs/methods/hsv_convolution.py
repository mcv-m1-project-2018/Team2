import numpy as np

from methods.operations import segregation
from methods.window import convolution


class hsv_convolution:

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask, regions = convolution(mask)

        return regions, mask, im
