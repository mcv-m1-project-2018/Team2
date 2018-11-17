from typing import List, Tuple

import cv2
import numpy as np
from functional import seq
from methods.operations.text import detect_text
from model import Picture
from model import Rectangle


class FLANN_Matcher:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array,Rectangle]]
    bf: cv2.BFMatcher
    sift: cv2.xfeatures2d.SIFT_create

    def __init__(self):
        self.db = []
        self.flann = cv2.FlannBasedMatcher_create()
        self.sift = cv2.xfeatures2d.SIFT_create(600)

    def query(self, picture: Picture) -> List[Picture]:
        mask ,rec = detect_text(picture.get_image())
        kp, des = self.sift.detectAndCompute(picture.get_image(), mask)
        FLANN_INDEX_KDTREE = 1
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        return (
            seq(self.db)
                .map(
                lambda p: (p[0], self.flann.knnMatch(np.asarray(p[2], np.float32), np.asarray(des, np.float32), 2)))
                .map(lambda p: (p[0], self._ratio_test(p[1])))
                .map(lambda p: (p[0], len(p[1])))
                .filter(lambda p: p[1] > 4)
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
            mask,bounding_text = detect_text(image.get_image())
            kp, des = self.sift.detectAndCompute(image.get_image(), mask)
            self.db.append((image, kp, des,bounding_text))
