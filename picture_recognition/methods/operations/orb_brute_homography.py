from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture


class ORBBruteHomography:

    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array]]
    bf: cv2.BFMatcher
    orb: cv2.ORB

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create()

    def query(self, picture: Picture) -> List[Picture]:
        kp, des = self.orb.detectAndCompute(picture.get_image(), None)

        return (
            seq(self.db)
            .map(lambda p: (p[0], self.bf.knnMatch(p[2], des, k=2)))
            .map(lambda p: (p[0], self._ratio_test(p[1])))
            .map(lambda p: (p[0], len(p[1])))
            .filter(lambda p: p[1] > 4)
            .sorted(lambda p: p[1], reverse=True)
            .map(lambda p: p[0])
            .take(10)
            .to_list()
        )

    def train(self, images: List[Picture]) -> None:
        for image in images:
            kp, des = self.orb.detectAndCompute(image.get_image(), None)
            self.db.append((image, kp, des))

    @staticmethod
    def _ratio_test(matches):
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        return good