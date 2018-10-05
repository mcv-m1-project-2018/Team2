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
        'hsv': _get_mask_hsv,
        'yuv': _get_mask_yuv
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    final_mask, result_seg = func(im)

    return final_mask


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
    
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
    #Colors extracted from the histograms
    
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([120, 255, 255])
    lower2_red=np.array([0,65,75])
    upper2_red=np.array([12,255,225])
    lower_red=np.array([240,65,6])
    upper_red=np.array([256,255,255])
    
    #Computing the mask
    red_mask = cv2.inRange(hsv_im, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_im, lower_blue, upper_blue)
   
    final_mask = red_mask + blue_mask 

    #Detection on the image  
    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg


def _get_mask_rgb(im):
    """
    get_mask_hsv(im,colorspace)
    
    Function to compute the color segmentation in RGB system
    
    Parameters   Value
   ----------------------
    'im'          Dataset image

    
    Returns a mask, a binary image with the color segmentation of the provided image
    and result_seg,an image with the detections done by the color segmentation(overlap between
    the provided image and the computed mask)
    """
    
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    #Colors extracted from the histograms
    upper_blue = np.array([0, 80, 255])
    lower_blue = np.array([0, 20, 100])
    lower_red=np.array([255,160,50])
    upper_red=np.array([220,0,10])
     
    #Computing the mask
    red_mask = cv2.inRange(rgb_im, lower_red, upper_red)
    blue_mask = cv2.inRange(rgb_im, lower_blue, upper_blue)
    red2_mask=cv2.inRange(rgb_im, lower2_red, upper2_red)
    final_mask = red_mask + blue_mask + red2_mask
    
    #Detection on the image    
    result_seg = cv2.bitwise_and(rgb_im, rgb_im, mask=final_mask)

    return final_mask, result_seg



