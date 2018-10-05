from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, histogram_equalization
from model import Data


class Method3:

    def train(self, data: List[Data]):
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        im = histogram_equalization(im, False)
        mask, im = segregation(im, 'hsv')
        mask = fill_holes(mask)
        mask = discard_geometry(mask)

        return mask


instance = Method3()
