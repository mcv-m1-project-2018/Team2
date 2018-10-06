from typing import List

import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, blur
from model import Data


class Method1:
    """
    Class Method1 
    
    In this class we implement the first detection method of the signals in the
    dataset images. In particular, in this first method the color segmentation
    is done in RGB.

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
        in RGB system
    
        Parameters   Value
       ----------------------
        'im'          Dataset image
    
        Returns the mask, binary image with the detections.
        """
        #Color segmentation in RGB
        mask, im = segregation(im, 'rgb')

        # Mask Blurring
        mask = blur(mask)
        
        #We apply a FLoodfill algorithm to the computed mask
        mask = fill_holes(mask)
        
        #Compute the final mask
        mask = discard_geometry.get_mask(mask)

        return mask, im


instance = Method1()
