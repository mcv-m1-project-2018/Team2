from enum import Enum

import numpy as np


class CompareHistogramsMethods(Enum):
    FULL_IMAGE = 0
    BLOCKS = 1


class CompareHistograms:

    def compare(self, hist: np.array, method: CompareHistogramsMethods, columns=-1):
        pass

    def train(self):
        pass
