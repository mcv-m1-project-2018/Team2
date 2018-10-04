import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation


def get_mask(im: np.array):
    # TODO histogram equalization with sliding windows
    mask, im = segregation(im, 'hsv')
    mask = fill_holes(mask)
    mask = discard_geometry(mask)

    return mask