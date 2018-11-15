from typing import List

from methods import AbstractMethod
from methods.operations import get_lines_rotation_and_crop
from model import Picture

import matplotlib.pyplot as plt

from methods.operations.text import detect_text


class w5(AbstractMethod):

    def __init__(self):
        pass

    def query(self, picture: Picture) -> List[Picture]:
        get_lines_rotation_and_crop(picture.get_image())
        return []

    def train(self, images: List[Picture]):
        for i in images:
            """im = cv2.resize(i.get_image(), (500, 500))
            top = cv2.morphologyEx(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, np.ones((35, 35), np.uint8))
            _, top = cv2.threshold(top, 200, 255, cv2.THRESH_BINARY)
            plt.subplot(1, 2, 1)
            plt.imshow(top, 'gray')

            plt.subplot(1, 2, 2)
            black = cv2.morphologyEx(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.MORPH_BLACKHAT,
                                   np.ones((35, 35), np.uint8))
            _, black = cv2.threshold(black, 200, 255, cv2.THRESH_BINARY)
            plt.imshow(black, 'gray')
            plt.show()"""

            im = detect_text(i.get_image())

            plt.imshow(im)
            plt.show()

            print(i.id)
        pass


instance = w5()
