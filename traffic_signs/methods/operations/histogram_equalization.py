import cv2

import numpy as np


def get_image(img: np.array, adaptive: bool):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel

    if not adaptive:
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
