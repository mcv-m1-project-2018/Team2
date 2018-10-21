from typing import List

import cv2
import numpy as np
from functional import seq

from methods.window import clear_non_region_mask
from model import Rectangle, Data
from matplotlib import pyplot as plt
from utils import get_filling_ratio


class DiscardGeometry:
    """Improves the mask by discarding invalid geometric shapes, such as those with too little area or those
    which are not "square" enough'"""

    def __init__(self):
        self.min_area = np.inf
        self.min_fill_ratio = np.inf
        self.max_fill_ratio = 0
        self.min_form_factor = np.inf
        self.max_form_factor = 0

    def train(self, data: List[Data]):
        """
        train(data)
        Performs a training of the operation.

        Parameters    Value
        ----------------------
        data          The training dataset
        """
        self.min_area, self.min_fill_ratio, self.max_fill_ratio, self.min_form_factor, self.max_form_factor = \
            seq(data).flat_map(lambda d: seq(d.gt).map(lambda gt: (d.get_mask_img(), gt)).to_list()) \
                     .map(lambda l: (l[1].get_area(), get_filling_ratio(l[1], l[0]), l[1].get_form_factor())) \
                     .reduce(lambda accum, l: (min(accum[0], l[0]), min(accum[1], l[1]), max(accum[2], l[1]),
                                               min(accum[3], l[2]), max(accum[4], l[2])),
                             (np.inf, np.inf, 0, np.inf, 0))

    def get_mask(self, mask: np.array, regions: List[Rectangle]):
        ret = []
        for region in regions:
            fr = get_filling_ratio(region, mask)
            ff = region.get_form_factor()
            if (region.get_area() < self.min_area or
                    fr < self.min_fill_ratio or
                    fr > self.max_fill_ratio or
                    ff < self.min_form_factor or
                    ff > self.max_form_factor):
                cv2.rectangle(mask, (region.top_left[1], region.top_left[0]),
                              (region.get_bottom_right()[1], region.get_bottom_right()[0]), 0, thickness=cv2.FILLED)
            else:
                ret.append(region)

        mask = clear_non_region_mask(mask, ret)

        return mask, ret


instance = DiscardGeometry()

