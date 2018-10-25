from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, histogram_equalization, blur
from model import Data


class Method4:
    """
    Class Method4 
    
    In this class we implement the fourth detection method of the signals in the
    dataset image. In particular, in this fourth method the color segmentation
    is done in HSV to an image with the Y-histogram adaptive equalised.

    """

    def train(self, data: List[Data]):
        """
        train(data)
    
        Function to train the values used in discard_geometry
    
        Parameters      Value
       ----------------------
        'data'          All the Data elements
        """
        discard_geometry.train(data)

    def get_mask(self, im: np.array):
        """
        get_mask(im)

        Function to compute the mask of an certain image,in this case is done
        in HSV system to an image with Y-histogram adaptive equalised

        Parameters    Value
       ----------------------
        'im'          Dataset image

        Returns the mask, binary image with the detections.
        """
        # Adaptive-equalization of the Y channel of the image
        im = histogram_equalization(im, True)

        # Color segmentation in HSV
        mask, im = segregation(im, 'hsv')

        # Mask Blurring
        mask = blur(mask)

        # Hole filling
        mask = fill_holes(mask)

        # Compute the final mask
        mask = discard_geometry.get_mask(mask)

        return mask, im


instance = Method4()
