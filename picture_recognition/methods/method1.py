from typing import List

import numpy as np

from picture_recognition.methods.operations import CompareHistograms, CompareHistogramsMethods, HistogramTypes
from picture_recognition.model import Picture


class Method1:
    compare_histograms: CompareHistograms

    def __init__(self):
        self.compare_histograms = CompareHistograms(CompareHistogramsMethods.FULL_IMAGE, HistogramTypes.HSV)

    def query(self, im: np.array) -> List[(Picture, float)]:
        return self.compare_histograms.query(im)

    def train(self, images: List[Picture]):
        self.compare_histograms.train(images)


instance = Method1
