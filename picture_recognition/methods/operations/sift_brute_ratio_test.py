from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture
from methods.operations.text import detect_text

from model import Rectangle


class SIFTBruteRatioTest:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array,Rectangle]]
    bf: cv2.BFMatcher
    sift: cv2.xfeatures2d.SIFT_create

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create()
        self.sift = cv2.xfeatures2d.SIFT_create(500)

    def query(self, picture: Picture) -> List[Picture]:
        mask,rec=detect_text(picture.get_image())
        kp, des = self.sift.detectAndCompute(picture.get_image(), mask)

        return (
            seq(self.db)
                .map(lambda p: (p[0], self.bf.knnMatch(p[2], des, k=2)))
                .map(lambda p: (p[0], self._ratio_test(p[1])))
                .map(lambda p: (p[0], len(p[1])))
                .filter(lambda p: p[1] > 0)
                .sorted(lambda p: p[1], reverse=True)
                .map(lambda p: p[0])
                .take(10)
                .to_list()
        )

    @staticmethod
    def _ratio_test(matches):
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def train(self, images: List[Picture]) -> None:
        for image in images:
            mask,bounding_text =detect_text(image.get_image())
            kp, des = self.sift.detectAndCompute(image.get_image(), mask)
            self.db.append((image, kp, des,bounding_text))
