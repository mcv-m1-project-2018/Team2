"""Calculate the mask using pixel segmentation and then filling the gaps"""
import cv2
from methods.operations import segregation
import numpy as np


def get_mask(mask: np.array):
    im_floodfill = mask.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = mask | im_floodfill_inv

    return im_out


if __name__ == '__main__':
    im = cv2.imread('../../datasets/train/00.000948.jpg')
    mask_hsv = segregation.get_mask(im, 'hsv')
    mask_hsv = get_mask(mask_hsv)
    cv2.imshow('image', im)
    cv2.imshow('mask', mask_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
