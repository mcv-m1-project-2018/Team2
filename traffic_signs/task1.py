from datasets import DatasetManager, Rectangle, GroundTruth
from typing import List
import cv2
import numpy as np


def get_mask_area(gt, mask_name):
    whites = 0
    mask = cv2.imread(mask_name)
    for i in range(gt.rectangle.top_left[0], gt.rectangle.get_bottom_right()[0]):
        for j in range(gt.rectangle.top_left[1], gt.rectangle.get_bottom_right()[1]):
            if mask[i][j] != 0:
                whites += 1
    return whites


def get_filling_factor(gt: GroundTruth, mask):
    # compute the area of bboxes
    bbox_area = gt.rectangle.get_area()
    mask_area = get_mask_area(gt, mask)

    # return the filling ratio
    return mask_area / bbox_area


class SignTypeStats:
    max_area: float = 0
    min_area: float = np.inf
    form_factor: List[float] = []
    filling_ratio: List[float] = []

    def add_sign(self, gt: GroundTruth, mask):
        self.max_area = max(self.max_area, gt.rectangle.get_area())
        self.min_area = min(self.min_area, gt.rectangle.get_area())
        self.form_factor.append(float(gt.rectangle.width / gt.rectangle.height))
        self.filling_ratio.append(get_filling_factor(gt, mask))

    def get_avg(self, data_length):
        return np.mean(self.form_factor), \
               np.mean(self.filling_ratio), \
               len(self.form_factor) / data_length


if __name__ == '__main__':
    dataManager = DatasetManager("../datasets/train")
    data = dataManager.data

    sign_type_stats = {}

    for sample in data:
        for gt in sample.gt:
            if gt.type in sign_type_stats.keys():
                sign_type_stats[gt.type] = SignTypeStats()

            sign_type_stats[gt.type].add_sign(gt, sample.get_mask_img())

    for key, value in sign_type_stats:
        print(key + ': ', value.get_avg())
