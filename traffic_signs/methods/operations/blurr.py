import cv2
import numpy as np



def mask_blurring(mask: np.array):
    """
    mask_blurring(mask)
    
    Function to blurr the resulting masks
    
    Parameters   Value
   ----------------------
    'mask'        Computed mask: Binary image with the signals detections obtained by the image segmentation 
    
    
    Returns the blurred mask 
    """
   
    mask_blurred=cv2.GaussianBlur(mask,(25,25),0)
    
    return mask_blurred
    
    