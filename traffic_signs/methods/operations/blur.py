import cv2
import numpy as np


def mask_blurring(mask: np.array):
    """
    mask_blurring(mask)
    
    Function to blur the resulting masks
    
    Parameters    Value
   ----------------------
    'mask'        Binary image with the signals detections obtained by the image segmentation
    
    
    Returns the blurred mask 
    """

    mask_blurred = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask_blurred
