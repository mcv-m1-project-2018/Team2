from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, histogram_equalization, blur
from model import Data


class Method3:

    def train(self, data: List[Data]):
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        im = histogram_equalization(im, False)
        mask, im = segregation(im, 'hsv')
        mask = blur(mask)
        mask = fill_holes(mask)
        mask = discard_geometry.get_mask(mask)

        return mask, im


instance = Method3()
