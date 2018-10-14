from typing import List

import cv2
import numpy as np
from functional import seq

from model import Rectangle, Data
from matplotlib import pyplot as plt
from utils import get_filling_ratio


class DiscardGeometry:
    """Improves the mask by discarding invalid geometric shapes, such as those with too little area or those
    which are not "square" enough'"""

    def __init__(self):
        self.min_area = np.inf
        self.min_fill_ratio = np.inf
        self.max_fill_ratio = 0
        self.min_form_factor = np.inf
        self.max_form_factor = 0

    def train(self, data: List[Data]):
        """
        train(data)
        Performs a training of the operation.

        Parameters    Value
        ----------------------
        data          The training dataset
        """
        self.min_area, self.min_fill_ratio, self.max_fill_ratio, self.min_form_factor, self.max_form_factor = \
            seq(data).flat_map(lambda d: seq(d.gt).map(lambda gt: (d.get_mask_img(), gt.rectangle)).to_list()) \
                     .map(lambda l: (l[1].get_area(), get_filling_ratio(l[1], l[0]), l[1].get_form_factor())) \
                     .reduce(lambda accum, l: (min(accum[0], l[0]), min(accum[1], l[1]), max(accum[2], l[1]),
                                               min(accum[3], l[2]), max(accum[4], l[2])),
                             (np.inf, np.inf, 0, np.inf, 0))

    def get_mask(self, mask: np.array):
        a, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region = []
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

            fr = get_filling_ratio(rectangle, mask)
            ff = rectangle.get_form_factor()
            if (rectangle.get_area() < self.min_area or
                    fr < self.min_fill_ratio or
                    fr > self.max_fill_ratio or
                    ff < self.min_form_factor or
                    ff > self.max_form_factor):
                cv2.rectangle(mask, (min_point[1], min_point[0]), (max_point[1], max_point[0]), 0, thickness=cv2.FILLED)
            else:
                region.append(rectangle)

        return mask, region


instance = DiscardGeometry()


def main():
    mask = cv2.imread('../datasets/train/mask/mask.00.000948.png', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    plt.imshow(DiscardGeometry().get_mask(mask), 'gray')
    plt.show()
