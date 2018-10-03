"""Calculate the mask by using pixel segmentation. The threshold values have been tuned manually"""
import cv2
import numpy as np


def get_mask(im: np.array):
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)

    lower_red = np.array([80, 150, 50])
    upper_red = np.array([180, 255, 255])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    red_mask = cv2.inRange(hsv_im, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_im, lower_blue, upper_blue)

    final_mask = red_mask + blue_mask

    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg
