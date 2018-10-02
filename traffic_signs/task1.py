from datasets import DatasetManager, Rectangle, GroundTruth
from typing import List
import numpy as np
import cv2


def get_mask_area(gt: GroundTruth, mask):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    gray_mask_cropped = gray_mask[
                        int(gt.rectangle.top_left[0]):int(gt.rectangle.get_bottom_right()[0]) + 1,
                        int(gt.rectangle.top_left[1]):int(gt.rectangle.get_bottom_right()[1]) + 1
                        ]
    _, img = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)


    whites = cv2.countNonZero(img)
    return whites


def get_filling_factor(gt: GroundTruth, mask: str):
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
    dataManager.load_data()
    data = dataManager.data

    sign_type_stats = {}

    print('Starting')
    i = 0
    for sample in data:
        print(str(i) + "/" + str(len(data)))
        print(len(sample.gt))
        i += 1
        mask = sample.get_mask_img()
        for gt in sample.gt:
            if gt.type not in sign_type_stats.keys():
                sign_type_stats[gt.type] = SignTypeStats()

            sign_type_stats[gt.type].add_sign(gt, mask)

    for key, value in sign_type_stats:
        print(key + ': ', value.get_avg())
