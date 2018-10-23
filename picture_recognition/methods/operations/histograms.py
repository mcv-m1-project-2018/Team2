import numpy as np
from enum import Enum
import cv2

class HistogramTypes(Enum):
    HSV = 0,
    LUV = 1


def get_histogram(im: np.array, histogramType=HistogramTypes.HSV) -> np.array:
    
    im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hist = []
    hist= cv2.calcHist([im], [0,1,2], None , [256], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

    pass

