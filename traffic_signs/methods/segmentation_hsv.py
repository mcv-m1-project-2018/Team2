from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, morpho
from model import Data
import matplotlib.pyplot as plt


class SegmentationHSV:
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
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        """
        get_mask(im)
    
        Function to compute the mask of an certain image,in this case is done 
        in HSV system
    
        Parameters   Value
       ----------------------
        'im'          Dataset image
    
        Returns the mask, binary image with the detections.
        """

        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        # Blurring the mask
        mask = morpho(mask)

        # Hole filling
        mask = fill_holes(mask)

        # Compute the final mask
        mask = discard_geometry.get_mask(mask)

        return mask, im


instance = SegmentationHSV()
