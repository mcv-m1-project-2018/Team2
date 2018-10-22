from typing import List

import numpy as np

from methods.operations import discard_geometry, get_cc_regions
from model import Data
from .operations import segregation, morpho, template_matching


class hsv_cc_template:

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)

        regions = get_cc_regions(mask)

        # Compute the final mask
        mask, regions = discard_geometry.get_mask(mask, regions)

        mask, regions = template_matching.template_matching_reg(mask, regions)

        return regions, mask, im

    def train(self, data: List[Data]):
        template_matching.train_masks(data)
        discard_geometry.train(data)
        pass


instance = hsv_cc_template()
