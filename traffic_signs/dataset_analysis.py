from dataset_manager import DatasetManager, Rectangle, GroundTruth
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tabulate import tabulate
from functional import seq


def get_cropped(gt: GroundTruth, img):
    img_cropped = img[
                  int(gt.rectangle.top_left[0]):int(gt.rectangle.get_bottom_right()[0]) + 1,
                  int(gt.rectangle.top_left[1]):int(gt.rectangle.get_bottom_right()[1]) + 1
                  ]
    return img_cropped


def get_mask_area(gt: GroundTruth, mask):
    mask_cropped = get_cropped(gt, mask)
    _, img = cv2.threshold(mask_cropped, 0, 255, cv2.THRESH_BINARY)

    whites = cv2.countNonZero(img)
    return whites


def get_histogram_RGB(img, mask):
    hist = np.zeros((3, 256))
    #  plt.subplot(121)
    #  plt.imshow(cv2.cvtColor(get_cropped(gt,img), cv2.COLOR_BGR2RGB))
    #  plt.subplot(122)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist[i] = cv2.calcHist([img], [i], mask, [256], [0, 256])[:, 0]

    #   plt.plot(hist[i], color=col)
    #   plt.xlim([0, 256])

    # plt.show()
    return hist


def get_histogram_gray(img):
    # TODO
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    return hist


def get_filling_factor(gt: GroundTruth, mask):
    # compute the area of bboxes
    bbox_area = gt.rectangle.get_area()
    mask_area = get_mask_area(gt, mask)

    # return the filling ratio
    return mask_area / bbox_area


class SignTypeStats:
    area: List[float]
    form_factor: List[float]
    filling_ratio: List[float]
    histogram: List[int]

    def __init__(self):
        self.area = []
        self.form_factor = []
        self.filling_ratio = []
        self.histogram = np.zeros((3, 256))

    def add_sign(self, gt: GroundTruth, img, mask):
        self.area.append(gt.rectangle.get_area())
        self.form_factor.append(float(gt.rectangle.width / gt.rectangle.height))
        self.filling_ratio.append(get_filling_factor(gt, mask))
        self.histogram = self.histogram + get_histogram_RGB(img, mask)

    def get_avg(self, data_length):
        return (max(self.area), min(self.area), np.mean(self.area), np.std(self.area)), \
               (max(self.form_factor), min(self.form_factor), np.mean(self.form_factor), np.std(self.form_factor)), \
               (max(self.filling_ratio), min(self.filling_ratio), np.mean(self.filling_ratio), np.std(self.filling_ratio)), \
                len(self.form_factor) / data_length


if __name__ == '__main__':
    dataManager = DatasetManager("../datasets/train")
    dataManager.load_data()
    data = dataManager.data

    sign_type_stats = {}

    print('Starting')
    total = 0
    for sample in data:
        img = sample.get_img()
        mask = sample.get_mask_img()
        for gt in sample.gt:
            if gt.type not in sign_type_stats.keys():
                sign_type_stats[gt.type] = SignTypeStats()

            sign_type_stats[gt.type].add_sign(gt, img, mask)
            total += 1


    for key, value in sign_type_stats.items():
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            plt.plot(value.histogram[i], color = col)
            plt.xlim([0, 256])
        plt.show()

    print(tabulate(seq(sign_type_stats.items())
          .order_by(lambda kv: ord(kv[0]))
          .map(lambda kv: list((kv[0],) + kv[1].get_avg(total) + (kv[1].max_area, kv[1].min_area)))
          .map(lambda l: seq(l).map(str).to_list())
          .reduce(lambda a, b: a + [b], []),
          ["Sign Type", "Avg. form factor", "Avg. fill ratio", 'Percentage', 'Max area', 'Min area']))

