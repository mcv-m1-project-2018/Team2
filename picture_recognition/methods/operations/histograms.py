import numpy as np
from enum import Enum


class HistogramTypes(Enum):
    HSV = 0,
    LUV = 1


def get_histogram(im: np.array, histogramType=HistogramTypes.HSV) -> np.array:
    # TODO
    pass
