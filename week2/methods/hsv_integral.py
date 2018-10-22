import cv2
from typing import List

import numpy as np

from methods.operations import segregation, fill_holes, morpho, discard_geometry
from methods.window import integral
from model import Data
import matplotlib.pyplot as plt

class hsv_integral:

    def train(self, data: List[Data]):
        pass

    def get_mask(self, im: np.array):
        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        mask = morpho(mask)

        mask, regions = integral(mask)

        return regions, mask, im


instance = hsv_integral()
