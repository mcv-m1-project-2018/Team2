import cv2
import numpy as np

from model import GroundTruth, Rectangle
from matplotlib import pyplot as plt


def get_filling_factor(rectangle: Rectangle, mask: np.array):
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
