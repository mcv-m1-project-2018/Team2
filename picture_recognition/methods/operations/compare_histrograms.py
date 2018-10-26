import math
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from functional import seq

from methods.operations.histograms import HistogramTypes, get_histogram
from model import Picture


class CompareHistogramsMethods(Enum):
    FULL_IMAGE = 0
    BLOCKS = 1


class CompareHistograms:
    method: CompareHistogramsMethods
    histogram_type: HistogramTypes
    db: List[Tuple[Picture, np.array]]

    def __init__(self, method: CompareHistogramsMethods, histogram_type: HistogramTypes):
        self.method = method
        self.histogram_type = histogram_type

    def query(self, picture: Picture) -> List[Tuple[Picture, float]]:
        hist = self._get_histogram(picture.get_image())
        return (
            seq(self.db)
                # Calculate histogram similarity
                .map(lambda entry: (entry[0], self._compare_histograms_full(hist, entry[1])))
                # Calculate distance to center
                .map(lambda entry: (entry[0], self._euclidean_distance_to_origin(entry[1])))
                # Order by distance
                .order_by(lambda entry_res: entry_res[1])
                # Take first 10
                .take(10)
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
                val.append(cv2.compareHist(h1[i][j], h2[i][j], cv2.HISTCMP_CORREL))
            val_list.append(val)

        return val_list

    def _compare_histograms_blocks(self, h1: List[np.array], h2: List[np.array]) -> List[np.array]:
        channels_range = range(0, 1)
        if self.histogram_type == HistogramTypes.HSV:
            channels_range = range(0, 1)
        elif self.histogram_type == HistogramTypes.YCbCr:
            channels_range = range(1, 3)
        val = []
        for block_num in range(len(h1)):
            val_blocks = []
            for channel in channels_range:
                val_blocks.append(cv2.compareHist(h1[block_num][channel], h2[block_num][channel], cv2.HISTCMP_CORREL))
            val.append(val_blocks)

        return val

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

    def _get_histogram(self, image: np.array, columns=1, rows=1) -> List[List[np.array]]:
        block_hist = []
        block_x = image.shape[0] / rows
        block_y = image.shape[1] / columns

        for i in range(rows):
            for j in range(columns):
                block = image[int(i * block_x):int((i + 1) * block_x),
                        int(j * block_y):int((j + 1) * block_y)]
                block_hist.append(get_histogram(block, self.histogram_type))

        return block_hist
