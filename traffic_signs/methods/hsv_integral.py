from typing import List

import numpy as np

from methods.operations import segregation, fill_holes, morpho, discard_geometry
from methods.window import integral
from model import Data


class hsv_integral:

    def train(self, data: List[Data]):
        pass

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)

        mask, regions = integral(mask)
        mask, regions = discard_geometry.get_mask(mask, regions)

        return regions, mask, im


instance = hsv_integral()
