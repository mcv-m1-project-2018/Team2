import cv2

import numpy as np


def get_image(img: np.array, adaptive: bool):
    """ 
    get_image(img, adaptive)
    
    Function to modify the dataset images by equalizing the histogram of the Y channel,the equalization 
    can be adaptive or not.
    
    Parameters   Value
   ----------------------
    'img'        Dataset image 
    
    'adaptive'   Boolean which indicates whether the equalization should be adaptive or not
    
    
    Returns the image with the equalized Y histogram 
    
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    if not adaptive:
        # Not adaptive equalization
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        # Adaptive equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
