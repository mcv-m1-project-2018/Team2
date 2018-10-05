import cv2
import numpy as np

def mask_blurring(mask: np.array):
    
    mask_blurred=cv2.GaussianBlur(mask,(25,25),0)
    
    return mask_blurred
    
    