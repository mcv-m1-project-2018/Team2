import math
from enum import Enum, IntEnum
from typing import List, Tuple

import cv2
import numpy as np
from functional import seq

from methods.operations.histograms import HistogramTypes, get_histogram
from model import Picture
import matplotlib.pyplot as plt

K = 10


class CompareHistogramsMethods(Enum):
    FULL_IMAGE = 0
    BLOCKS_16_16 = 1
    BLOCKS_32_32 = 2
    BLOCKS_4_4 = 3


class HistogramComparisonMethods(IntEnum):
    HISTCMP_CORREL = cv2.HISTCMP_CORREL
    HISTCMP_HELLINGER = cv2.HISTCMP_HELLINGER


class CompareHistograms:
    method: CompareHistogramsMethods
    histogram_type: HistogramTypes
    db: List[Tuple[Picture, np.array]]
    histogram_comparison_method: int

    def __init__(self, method: CompareHistogramsMethods, histogram_type: HistogramTypes,
                 histogram_comparison_method=HistogramComparisonMethods.HISTCMP_CORREL):
        self.method = method
        self.histogram_type = histogram_type
        self.histogram_comparison_method = histogram_comparison_method

    def query(self, picture: Picture) -> List[Picture]:
        hist = self._get_histogram(picture.get_image())
        return (
            seq(self.db)
                # Calculate histogram similarity
                .map(lambda entry: (entry[0], self._compare_histograms_full(hist, entry[1])))
                # Calculate distance to center
                .map(lambda entry: (entry[0], self._euclidean_distance_to_origin(entry[1])))
                # Order by distance
                .sorted(lambda entry_res: entry_res[1],
                        False if self.histogram_comparison_method == cv2.HISTCMP_HELLINGER else True)
                .map(lambda entry: entry[0])
                # Take first K
                .take(K)
                .to_list()
        )

    def _compare_histograms_full(self, h1: List[np.array], h2: List[np.array]) -> List[List[float]]:
        channels_range = range(0, 1)
        if self.histogram_type == HistogramTypes.HSV:
            channels_range = range(0, 1)
        elif self.histogram_type == HistogramTypes.YCbCr:
            channels_range = range(1, 3)

        val_list = []
        # Iterate list of blocks
        for i in range(len(h1)):
            val = []
            # Iterate channels to compare
            for j in channels_range:
                val.append(cv2.compareHist(h1[i][j], h2[i][j], self.histogram_comparison_method))
            val_list.append(val)

        return val_list

    @staticmethod
    def _euclidean_distance_to_origin(pos: List[List[float]]) -> float:
        total = 0
        for i in range(len(pos)):
            res: float = 0
            for j in range(len(pos[i])):
                res += pos[i][j] ** 2
            total += math.sqrt(res)

        return total / len(pos)

    def train(self, images: List[Picture]) -> None:
        self.db = []
        for image in images:
            self.db.append((image, self._get_histogram(image.get_image())))

    def _get_histogram(self, image: np.array) -> List[List[np.array]]:
        columns = rows = 1
        if self.method == CompareHistogramsMethods.FULL_IMAGE:
            columns = rows = 1
        elif self.method == CompareHistogramsMethods.BLOCKS_16_16:
            columns = rows = 16
        elif self.method == CompareHistogramsMethods.BLOCKS_32_32:
            columns = rows = 32
        elif self.method == CompareHistogramsMethods.BLOCKS_4_4:
            columns = rows = 4
        block_hist = []
        block_x = image.shape[0] / rows
        block_y = image.shape[1] / columns

        """plt.figure()
        pos = 1"""
        for i in range(rows):
            for j in range(columns):
                block = image[int(i * block_x):int((i + 1) * block_x),
                        int(j * block_y):int((j + 1) * block_y)]
                block_hist.append(get_histogram(block, self.histogram_type))
                """plt.subplot(4, 4, pos)
                pos += 1
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))

        plt.show()"""
        return block_hist
