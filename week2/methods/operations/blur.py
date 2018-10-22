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

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    return mask
