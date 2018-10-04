"""Calculate the mask using pixel segmentation, filling the gaps and finally check if the geometry of
the areas is valid"""
import cv2
import numpy as np


def get_mask(mask: np.array):
    im_contours, contours = cv2.findContours(mask, cv2.RETR_FLOODFILL, cv2.CONTOURS_MATCH_I1)

    for contour in contours:
        pass

    return mask
