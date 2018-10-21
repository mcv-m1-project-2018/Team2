from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, morpho
from methods.window import sliding_window
from model import Data
import matplotlib.pyplot as plt


class HSV_SW_TM:
    """
    
    In this class we implement the first detection method of the signals in the
    dataset images. In particular, in this first method the color segmentation
    is done in HSV.

    """

    def train(self, data: List[Data]):
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)

        mask, regions = sliding_window(mask)

        mask, regions = discard_geometry.get_mask(mask, regions)

        return regions, mask, im


instance = HSV_SW_TM()
