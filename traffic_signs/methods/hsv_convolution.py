from typing import List

import numpy as np

from methods.operations import segregation, fill_holes, morpho
from methods.window import convolution
from model import Data


class hsv_convolution:

    def train(self, data: List[Data]):
        pass

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)
        mask = fill_holes(mask)

        mask, regions = convolution(mask)
        return regions, mask, im


instance = hsv_convolution()
