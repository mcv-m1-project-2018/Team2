from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, histogram_equalization, blurring
from model import Data


class Method2:

    def train(self, data: List[Data]):
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        mask = segregation(im, 'hsv')
        mask = fill_holes(mask)
        mask=blurring(mask)
        #mask = discard_geometry.get_mask(mask)

        return mask


instance = Method2()
