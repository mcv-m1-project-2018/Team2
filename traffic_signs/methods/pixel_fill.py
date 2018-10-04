"""Calculate the mask using pixel segmentation and then filling the gaps"""
import cv2
from methods import pixel
import numpy as np


def get_mask(im: np.array, color_space):
    mask, im_cropped = pixel.get_mask(im,color_space)

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
    mask = get_mask(im,color_space)
    cv2.imshow('image', im)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
