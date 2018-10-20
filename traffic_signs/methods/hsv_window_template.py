from typing import List

import numpy as np

from .operations import segregation, fill_holes, morpho, template_matching
from .window import move_window
from model import Data


class hsv_window_template:

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)
        mask = fill_holes(mask)

        mask, regions = move_window(mask)

        res = template_matching.template_matching(mask)

        return regions, mask, im

    def train(self, data: List[Data]):
        template_matching.train_masks(data)
        pass


instance = hsv_window_template()
