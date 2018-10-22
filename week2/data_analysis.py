from model import DatasetManager
from typing import List
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})
from functional import seq

from model import GroundTruth
# from methods.operations import histogram_equalization
from utils import get_filling_ratio, get_histogram, print_all_histograms


class SignTypeStats:
    area: List[float]
    form_factor: List[float]
    filling_ratio: List[float]
    histogram: np.array

    def __init__(self):
        self.area = []
        self.form_factor = []
        self.filling_ratio = []
        self.histogram = np.zeros((256, 2, 3))

    def add_sign(self, gt: GroundTruth, img: np.array, mask: np.array):
        self.area.append(gt.rectangle.get_area())
        self.form_factor.append(float(gt.rectangle.width / gt.rectangle.height))
        self.filling_ratio.append(get_filling_ratio(gt.rectangle, mask))
        hists_rgb = get_histogram(img, gt, mask, False)
        hists_hsv = get_histogram(img, gt, mask, True)
        for i in range(3):
            self.histogram[:, 0, i] += hists_rgb[i][:, 0]
            self.histogram[:, 1, i] += hists_hsv[i][:, 0]

    def get_avg(self, data_length):
        return (max(self.area), min(self.area), np.mean(self.area), np.std(self.area)), \
               (max(self.form_factor), min(self.form_factor), np.mean(self.form_factor), np.std(self.form_factor)), \
               (max(self.filling_ratio), min(self.filling_ratio), np.mean(self.filling_ratio),
                np.std(self.filling_ratio)), \
               len(self.form_factor) / data_length


def get_signs_stats(data):
    sign_type_stats = {}
    total = 0
    for sample in data:
        img = sample.get_img()
        mask = sample.get_mask_img()

        for gt in sample.gt:
            if gt.type not in sign_type_stats.keys():
                sign_type_stats[gt.type] = SignTypeStats()

            sign_type_stats[gt.type].add_sign(gt, img, mask)
            total += 1

    return sign_type_stats, total


def split_analysis(data):
    sign_type_stats, total = get_signs_stats(data)
    print_all_histograms(sign_type_stats)


def data_analysis(data):
    sign_type_stats, total = get_signs_stats(data)

    print_all_histograms(sign_type_stats)

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

        mins = np.array(seq(stat_data[i + 1]).map(lambda v: v[0]).to_list())
        maxes = np.array(seq(stat_data[i + 1]).map(lambda v: v[1]).to_list())
        means = np.array(seq(stat_data[i + 1]).map(lambda v: v[2]).to_list())
        stds = np.array(seq(stat_data[i + 1]).map(lambda v: v[3]).to_list())
        plt.xticks(np.arange(len(stat_data[0])), stat_data[0])
        plt.errorbar(np.arange(len(mins)), means, yerr=stds, fmt='ok', lw=3, capsize=5)
        plt.errorbar(np.arange(len(mins)), means, yerr=[means - mins, maxes - means],
                     fmt='.k', ecolor='gray', lw=1, capsize=3)
    plt.show()


if __name__ == '__main__':
    # Analysis

    dataManager = DatasetManager("../datasets/train")
    print('Loading data...')
    dataManager.load_data()
    data = dataManager.data
    data_analysis(data)
