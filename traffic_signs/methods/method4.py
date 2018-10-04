import numpy as np
from methods.operations import fill_holes, discard_geometry, segregation, histogram_equalization


def get_mask(im: np.array):
    im = histogram_equalization(im, True)
    mask, im = segregation(im, 'hsv')
    mask = fill_holes(mask)
    mask = discard_geometry(mask)

    return mask
