"""Calculate the mask by using pixel segmentation. The threshold values have been tuned manually"""
import cv2
import numpy as np


def get_mask(im: np.array, color_space: str):
    switcher = {
        'rgb': _get_mask_rgb,
        'hsv': _get_mask_hsv,
        'yuv': _get_mask_yuv
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    final_mask, result_seg = func(im)

    return final_mask, result_seg


def _get_mask_hsv(im):
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


def _get_mask_rgb(im):
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    upper_red = np.array([255, 56, 50])
    lower_red = np.array([155, 15, 15])
    upper_blue = np.array([0, 80, 255])
    lower_blue = np.array([0, 20, 100])

    red_mask = cv2.inRange(rgb_im, lower_red, upper_red)
    blue_mask = cv2.inRange(rgb_im, lower_blue, upper_blue)
    final_mask = red_mask + blue_mask

    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg


def _get_mask_yuv(im):
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    yuv_im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    upper_red = np.array([])
    lower_red = np.array([])
    upper_blue = np.array([])
    lower_blue = np.array([])

    red_mask = cv2.inRange(yuv_im, lower_red, upper_red)
    blue_mask = cv2.inRange(yuv_im, lower_blue, upper_blue)
    final_mask = red_mask + blue_mask

    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg