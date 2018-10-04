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


def get_histogram_RGB(img, mask, prev_hist):
    #  plt.subplot(121)
    # plt.imshow(cv2.cvtColor(get_cropped(gt, img), cv2.COLOR_BGR2RGB))
    # plt.subplot(122)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
        # plt.plot(hist.ravel(), color=col)
        # plt.xlim([0, 256])
        prev_hist[:, :, i] += hist
    # plt.show()
    return


"""def get_histogram_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    return hist """

def get_histogram_equalization(img, adaptive):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel

    if adaptive is False:
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        return  cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


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
    histogram: np.array

    def __init__(self):
        self.area = []
        self.form_factor = []
        self.filling_ratio = []
        self.histogram = np.zeros((256, 1, 3))

    def add_sign(self, gt: GroundTruth, img, mask):
        self.area.append(gt.rectangle.get_area())
        self.form_factor.append(float(gt.rectangle.width / gt.rectangle.height))
        self.filling_ratio.append(get_filling_factor(gt, mask))
        get_histogram_RGB(img, mask, self.histogram)

    def get_avg(self, data_length):
        return (max(self.area), min(self.area), np.mean(self.area), np.std(self.area)), \
               (max(self.form_factor), min(self.form_factor), np.mean(self.form_factor), np.std(self.form_factor)), \
               (max(self.filling_ratio), min(self.filling_ratio), np.mean(self.filling_ratio),
                np.std(self.filling_ratio)), \
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

        if total < 10:
            plt.figure()
            plt.title('Histogram equalization')
            plt.subplot(131)
            plt.title('Original')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot(132)
            plt.title('Hist eq')
            plt.imshow(cv2.cvtColor(get_histogram_equalization(img, False), cv2.COLOR_BGR2RGB))
            plt.subplot(133)
            plt.title('Adaptive hist eq')
            plt.imshow(cv2.cvtColor(get_histogram_equalization(img, True), cv2.COLOR_BGR2RGB))

        for gt in sample.gt:
            if gt.type not in sign_type_stats.keys():
                sign_type_stats[gt.type] = SignTypeStats()

            sign_type_stats[gt.type].add_sign(gt, img, mask)
            total += 1

    plt.figure()
    subplt = 231
    for sign_type, stat in seq(sign_type_stats.items()).order_by(lambda kv: ord(kv[0])):
        plt.title('Histograms by sign type')
        color = ('b', 'g', 'r')
        plt.subplot(subplt)
        subplt += 1
        plt.title(sign_type)
        for i, col in enumerate(color):
            plt.plot(stat.histogram[:, :, i].ravel(), color=col)
            plt.xlim([0, 256])

    stat_data = seq(sign_type_stats.items()) \
        .order_by(lambda kv: ord(kv[0])) \
        .map(lambda kv: (kv[0],) + kv[1].get_avg(total)) \
        .reduce(lambda a, b: [a[0] + [b[0]], a[1] + [b[1]], a[2] + [b[2]], a[3] + [b[3]]], [[], [], [], []]) \
        .to_list()
    fields = ['Area', 'Form factor', 'Filling ratio']
    plt.figure()
    for i, field in enumerate(fields):
        plt.subplot(131 + i)
        plt.title(field)

        mins = np.array(seq(stat_data[i+1]).map(lambda v: v[0]).to_list())
        maxes = np.array(seq(stat_data[i+1]).map(lambda v: v[1]).to_list())
        means = np.array(seq(stat_data[i+1]).map(lambda v: v[2]).to_list())
        stds = np.array(seq(stat_data[i+1]).map(lambda v: v[3]).to_list())
        plt.xticks(np.arange(len(stat_data[0])), stat_data[0])
        plt.errorbar(np.arange(len(mins)), means, yerr=stds, fmt='ok', lw=3, capsize=5)
        plt.errorbar(np.arange(len(mins)), means, yerr=[means - mins, maxes - means],
                     fmt='.k', ecolor='gray', lw=1, capsize=3)

    plt.show()
