import cv2
import numpy as np
from matplotlib import pyplot as plt


def morphology_operations(mask: np.array, kernel=(3, 3)):
    """
    morphology_operation(mask)
    
    Function to apply morphological operatations (Opening and Closing) to the resulting masks
    
    Parameters    Value
   ----------------------
    'mask'        Binary image with the signals detections obtained by the image segmentation
    
    
    Returns the modified mask 
    """
    kernel_mat = np.ones(kernel, np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mat)

    mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_mat)

    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    return mask
