import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_lines_rotation_and_crop(im: np.array):
    resized = cv2.resize(im, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # uses the above two partial derivatives
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_gradientx = cv2.convertScaleAbs(sobelx)
    abs_gradienty = cv2.convertScaleAbs(sobely)
    # combine the two in equal proportions
    gray = cv2.addWeighted(abs_gradientx, 0.5, abs_gradienty, 0.5, 0)

    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    gray = cv2.Canny(gray, threshold1=0, threshold2=50, apertureSize=3)

    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    # lines = cv2.HoughLines(gray, 1, np.pi / 180, 150)
    i = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(i, (x1, y1), (x2, y2), (0, 255, 0), 2)

    """for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(i, (x1, y1), (x2, y2), (0, 0, 255), 2)"""

    plt.imshow(i)
    plt.show()
