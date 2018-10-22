import cv2
import numpy as np

from model import GroundTruth, Rectangle
from matplotlib import pyplot as plt
from functional import seq

plt.rcParams.update({'font.size': 16})


def get_histogram(img: np.array, gt: GroundTruth, mask: np.array, HVS):
    #  plt.subplot(121)
    # plt.imshow(cv2.cvtColor(get_cropped(gt, img), cv2.COLOR_BGR2RGB))
    # plt.subplot(122)
    if HVS is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = get_cropped(gt.rectangle, img)
    mask = get_cropped(gt.rectangle, mask)
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
        # plt.plot(hist.ravel(), color=col)
        # plt.xlim([0, 256])
    # plt.show()
    return hists


def print_all_histograms(sign_type_stats):
    for x in range(2):
        subplt = 231
        plt.figure()
        for sign_type, stat in seq(sign_type_stats.items()).order_by(lambda kv: ord(kv[0])):
            # plt.title('Histograms by sign type')
            color = ('b', 'g', 'r')
            ax = plt.subplot(subplt)
            subplt += 1
            plt.title(sign_type)

            for i, col in enumerate(color):
                plt.plot(stat.histogram[:, x, i].ravel(), color=col)
                plt.xlim([0, 256])

        if x == 0:

            ax.plot(stat.histogram[:, x, 2], '--r', label='Red')
            ax.plot(stat.histogram[:, x, 1], '--g', label='Green')
            ax.plot(stat.histogram[:, x, 0], '--b', label='Blue')
        else:
            ax.plot(stat.histogram[:, x, 2], '--r', label='H')
            ax.plot(stat.histogram[:, x, 1], '--g', label='S')
            ax.plot(stat.histogram[:, x, 0], '--b', label='V')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=True, ncol=3)


def get_filling_ratio(rectangle: Rectangle, mask: np.array):
    # compute the area of bboxes
    bbox_area = rectangle.get_area()
    whites = count_whites(rectangle, mask)

    # return the filling ratio
    return whites / bbox_area


def count_whites(rectangle: Rectangle, mask):
    mask_cropped = get_cropped(rectangle, mask)
    _, img = cv2.threshold(mask_cropped, 0, 255, cv2.THRESH_BINARY)

    whites = cv2.countNonZero(img)
    return whites


def get_cropped(rectangle: Rectangle, img):
    img_cropped = img[
                  int(rectangle.top_left[0]):int(rectangle.get_bottom_right()[0]) + 1,
                  int(rectangle.top_left[1]):int(rectangle.get_bottom_right()[1]) + 1
                  ]
    return img_cropped
