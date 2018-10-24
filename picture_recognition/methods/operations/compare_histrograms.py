import math

import cv2
from enum import Enum
from typing import List, Tuple

import numpy as np
from functional import seq

from picture_recognition.methods.operations.histograms import HistogramTypes, get_histogram
from picture_recognition.model import Picture


class CompareHistogramsMethods(Enum):
    FULL_IMAGE = 0
    BLOCKS = 1


class CompareHistograms:
    method: CompareHistogramsMethods
    histogram_type: HistogramTypes

    db: List[(Picture, np.array)]

    def __init__(self, method: CompareHistogramsMethods, histogram_type: HistogramTypes):
        self.method = method
        self.histogram_type = histogram_type

    def query(self, im: np.array) -> List[(Picture, float)]:

        if self.method == CompareHistogramsMethods.FULL_IMAGE:
            hist = get_histogram(im, self.histogram_type)
            return (
                seq(self.db)
                # Calculate histogram similarity
                .map(lambda entry: (entry[0], self._compare_histograms_full(hist, entry[1])))
                # Calculate distance to center
                .map(lambda entry: (entry[0], self._euclidean_distance([0 for _ in range(len(entry[1]))], entry[1])))
                # Order by distance
                .order_by(lambda entry_res: entry_res[1])
                # Take first 10
                .take(10)
                .to_list()
            )

        elif self.method == CompareHistogramsMethods.BLOCKS:
            pass

    def _compare_histograms_full(self, h1: np.array, h2: np.array) -> List[float]:
        channels = 1
        if self.histogram_type == HistogramTypes.HSV:
            channels = 1
        elif self.histogram_type == HistogramTypes.LUV:
            channels = 2

        val = []
        for i in range(channels):
            val.append(cv2.compareHist(h1[i], h2[i], cv2.HISTCMP_CORREL))
        return val

    @staticmethod
    def _euclidean_distance(pos1: List[float], pos2: List[float]) -> float:
        res = 0
        for i in range(len(pos1)):
            res += (pos1[i] - pos2[i]) ** 2
        return math.sqrt(res)

    def train(self, images: List[Picture]):
        self.db = []
        for image in images:
            self.db.append((image, get_histogram(image, self.histogram_type)))
