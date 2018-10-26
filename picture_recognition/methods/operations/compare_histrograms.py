import math
from enum import Enum
from typing import List

import cv2
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
    block_rows:int
    block_columns:int
    db: List[(Picture, np.array)]

    def __init__(self, method: CompareHistogramsMethods, histogram_type: HistogramTypes):
        self.method = method
        self.histogram_type = histogram_type
        self.block_rows=3
        self.block_columns=3

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
            
            return(
                
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
            
            
            pass

    def _compare_histograms_full(self, h1: np.array, h2: np.array) -> List[float]:
        channels_range = range(0, 1)
        if self.histogram_type == HistogramTypes.HSV:
            channels_range = range(0, 1)
        elif self.histogram_type == HistogramTypes.YCbCr:
            channels_range = range(1, 3)

        val = []
        for i in channels_range:
            val.append(cv2.compareHist(h1[i], h2[i], cv2.HISTCMP_CORREL))
        return val
    def _compare_histograms_blocks(self, h1:List[np.array], h2:List[np.array]) -> List[np.array]:
        
        channels_range = range(0, 1)
        if self.histogram_type == HistogramTypes.HSV:
            channels_range = range(0, 1)
        elif self.histogram_type == HistogramTypes.YCbCr:
            channels_range = range(1, 3)
        val_blocks=[]    
        val = []
        for i in channels_range:
            for j in range(len(h1[i])):
                val_blocks.append(cv2.compareHist(h1[i][j], h2[i][j], cv2.HISTCMP_CORREL))
            val.append(val_blocks)

        return val
    
    @staticmethod
    def _euclidean_distance(pos1: List[float], pos2: List[float]) -> float:
        res = 0
        for i in range(len(pos1)):
            res += (pos1[i] - pos2[i]) ** 2
        return math.sqrt(res)

    def train(self, images: List[Picture]):
        self.db = []
        if self.method == CompareHistogramsMethods.FULL_IMAGE:
            for image in images:
                self.db.append((image, get_histogram(image, self.histogram_type)))
        elif self.method == CompareHistogramsMethods.BLOCKS:
            block=[]
            for image in images:
                block_hist=[]
                block_x = im.shape[0] / self.block_rows
                block_y = im.shape[1] / self.block_columns
                
                if( j != self.block_columns and i != self.block_rows):
                    for i in range(self.block_rows):
                        for j in range(self.block_columns):
                            block = img[int(i * block_x):int((i + 1) *  block_x),int(j * block_y):int((j + 1) * block_y)]
                            block_hist.append(get_histogram(block, self.histogram_type))
                self.db.append((image,block_hist))
            pass
