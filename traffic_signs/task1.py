from datasets import DatasetManager, Size
from typing import List
import cv2
import numpy as np


def get_mask_area(gt, mask_name):
    # TODO
    mask = cv2.imread(mask_name)
    return 0


def get_filling_factor(gt, mask):

    # compute the area of bboxes
    bbox_area = (gt.bottom_right[0] - gt.top_left[0] + 1) * (gt.bottom_right[1] - gt.top_left[1] + 1)
    mask_area = get_mask_area(gt, mask)

    # return the filling ratio
    return mask_area/float(bbox_area)


class SignType:
    max_size: Size
    min_size: Size
    form_factor: List[float]
    filling_ratio: List[float]
    form_factor_avg: float
    filling_ratio_avg: float
    appearance_frequency: float

    def add_signal(self, gt , mask):
        self.max_size = self.max_size.max(gt.size)
        self.min_size = self.min_size.min(gt.size)
        self.form_factor.append(float(gt.size.width/gt.size.height))
        self.filling_ratio.append(get_filling_factor(gt, mask))

    def get_avg(self,data_length):
        self.form_factor_avg = np.mean(self.form_factor)
        self.filling_ratio_avg = np.mean(self.filling_ratio)
        self.appearance_frequency = len(self.form_factor)/data_length


if __name__ == '__main__':
    # TODO
    dataManager = DatasetManager("directorio")
    data = dataManager.data

    sign_types = {}

    for sample in data:
        for gt in sample.gt:
            if gt.type in sign_types.keys():
                pass
            else:
                pass

    pass
