"""Calculate the mask using pixel segmentation, filling the gaps and finally check if the geometry of
the areas is valid"""
from typing import List

import cv2
import numpy as np
from functional import seq

from model import Rectangle, Data
from matplotlib import pyplot as plt
from utils import get_filling_factor


class DiscardGeometry:
    PADDING_FACTOR = 0.8

    def __init__(self):
        self.min_area = np.inf
        self.min_fill_factor = np.inf
        self.max_fill_factor = 0

    def train(self, data: List[Data]):
        self.min_area, self.min_fill_factor, self.max_fill_factor = seq(data) \
            .flat_map(lambda d: seq(d.gt).map(lambda gt: (d.get_mask_img(), gt.rectangle)).to_list()) \
            .map(lambda l: (l[1].get_area(), get_filling_factor(l[1], l[0]))) \
            .reduce(lambda accum, l: (min(accum[0], l[0]), min(accum[1], l[1]), max(accum[2], l[1])),
                    (np.inf, np.inf, 0))

    def get_mask(self, mask: np.array):
        a, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            min_point = np.full(2, np.iinfo(np.int).max)
            max_point = np.zeros(2).astype(int)
            for point in contour:
                min_point[0] = min(min_point[0], int(point[0][1]))
                min_point[1] = min(min_point[1], int(point[0][0]))
                max_point[0] = max(max_point[0], int(point[0][1]))
                max_point[1] = max(max_point[1], int(point[0][0]))

            rectangle = Rectangle()
            rectangle.top_left = min_point.astype(int).tolist()
            rectangle.height = int(max_point[0] - min_point[0])
            rectangle.width = int(max_point[1] - min_point[1])

            ff = get_filling_factor(rectangle, mask)
            # cv2.rectangle(mask, tuple(min_point.astype(int)), tuple(max_point.astype(int)), 255, thickness=1)
            if (rectangle.get_area() < self.min_area * self.PADDING_FACTOR or
                    ff < self.min_fill_factor * self.PADDING_FACTOR or
                    ff > self.max_fill_factor / self.PADDING_FACTOR):
                cv2.rectangle(mask, (min_point[1], min_point[0]), (max_point[1], max_point[0]), 0, thickness=cv2.FILLED)

        return mask


instance = DiscardGeometry()


def main():
    mask = cv2.imread('../datasets/train/mask/mask.00.000948.png', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    plt.imshow(DiscardGeometry().get_mask(mask), 'gray')
    plt.show()
