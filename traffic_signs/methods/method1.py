from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation
from model import Data


class Method1:

    def train(self, data: List[Data]):
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        mask, im = segregation(im, 'rgb')
        mask = fill_holes(mask)
        mask = discard_geometry.get_mask(mask)

        return mask


instance = Method1()
