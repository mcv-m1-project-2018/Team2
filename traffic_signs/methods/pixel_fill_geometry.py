"""Calculate the mask using pixel segmentation, filling the gaps and finally check if the geometry of
the areas is valid"""
import cv2
from methods.pixel_fill import get_mask as get_mask_pixel_fill
import numpy as np


def get_mask(im: np.array):
    mask = get_mask_pixel_fill(im)
    pass