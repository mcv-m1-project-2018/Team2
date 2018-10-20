import cv2
import numpy as np
from . import fill_holes


def morphology_operations(mask: np.array, kernel_open=(3, 3), kernel_close=(11, 11)):
    """
    morphology_operation(mask)
    
    Function to apply morphological operatations (Opening and Closing) to the resulting masks
    
    Parameters    Value
   ----------------------
    'mask'        Binary image with the signals detections obtained by the image segmentation
    
    
    Returns the modified mask 
    """
    kernel_open_mat = np.ones(kernel_open, np.uint8)
    kernel_close_mat = np.ones(kernel_close, np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_mat)
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open_mat)

    return mask
