"""Calculate the mask by using pixel segmentation. The threshold values have been tuned manually"""
import cv2
import numpy as np


def get_mask(im: np.array, color_space: str):
    """
    get_mask(im,colorspace)
    
    Function to compute the color segmentation in a certain color space
    
    Parameters   Value
   ----------------------
    'im'          Dataset image
    
    'color_space' Colorspace of the color segmentation 
    
    Returns a mask, a binary image with the color segmentation of the provided image
    """

    switcher = {
        'rgb': _get_mask_rgb,
        'hsv': _get_mask_hsv
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    final_mask, result_seg = func(im)

    return final_mask, result_seg


def _get_mask_hsv(im):
    """
    get_mask_hsv(im,colorspace)

    Function to compute the color segmentation in HSV system

    Parameters   Value
   ----------------------
    'im'          Dataset image


    Returns a mask, a binary image with the color segmentation of the provided image
    and result_seg,an image with the detections done by the color segmentation(overlap between
    the provided image and the computed mask)
    """
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_red_begin = np.array([0, 150, 50])
    upper_red_begin = np.array([20, 255, 255])
    lower_red_end = np.array([160, 150, 50])
    upper_red_end = np.array([180, 255, 255])
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    # lower_white=np.array([0, 0, 225])
    # upper_white=np.array([145, 60, 255])

    red_mask_begin = cv2.inRange(hsv_im, lower_red_begin, upper_red_begin)
    red_mask_end = cv2.inRange(hsv_im, lower_red_end, upper_red_end)
    blue_mask = cv2.inRange(hsv_im, lower_blue, upper_blue)
    # white_mask= cv2.inRange(hsv_im, lower_white, upper_white)
    final_mask = red_mask_begin + red_mask_end + blue_mask
    # white_mask

    result_seg = cv2.bitwise_and(im, im, mask=final_mask)

    return final_mask, result_seg


def _get_mask_rgb(im):
    """
    get_mask_hsv(im, colorspace)

    Function to compute the color segmentation in RGB system

    Parameters   Value
   ----------------------
    'im'          Dataset image


    Returns a mask, a binary image with the color segmentation of the provided image
    and result_seg,an image with the detections done by the color segmentation(overlap between
    the provided image and the computed mask)
    """
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    upper_blue = np.array([0, 80, 255])
    lower_blue = np.array([0, 20, 100])
    lower2_red = np.array([0, 150, 50])
    upper2_red = np.array([15, 255, 225])
    lower_red = np.array([170, 160, 50])
    upper_red = np.array([180, 255, 255])

    red_mask = cv2.inRange(rgb_im, lower_red, upper_red)
    blue_mask = cv2.inRange(rgb_im, lower_blue, upper_blue)
    red2_mask = cv2.inRange(rgb_im, lower2_red, upper2_red)
    final_mask = red_mask + blue_mask + red2_mask

    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg
