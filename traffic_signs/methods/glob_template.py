from typing import List

import numpy as np

from methods.operations import discard_geometry, segregation, morpho, get_cc_regions, template_matching
from model import Data


class glob_template:
    """
    
    In this class we implement the first detection method of the signals in the
    dataset images. In particular, in this first method the color segmentation
    is done in HSV.

    """

    def train(self, data: List[Data]):
        """
        train(data)
    
        Function to train the values used in discard_geometry
    
        Parameters   Value
       ----------------------
        'data'          All the Data elements
        """
        template_matching.train_masks(data)

    def get_mask(self, im: np.array):
        template_matching.template_matching_global(im)

        return regions, mask, im


instance = glob_template()
