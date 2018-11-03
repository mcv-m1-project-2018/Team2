from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture




class FLANN_Matcher:

    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array]]
    bf: cv2.BFMatcher
    sift: cv2.xfeatures2d.SIFT_create

    def __init__(self):
        self.db = []
        self.flann = cv2.FlannBasedMatcher_create()
        self.sift = cv2.xfeatures2d.SIFT_create()

    def query(self, picture: Picture) -> List[Picture]:
        kp, des = self.sift.detectAndCompute(picture.get_image(), None)
        FLANN_INDEX_KDTREE = 1
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        return (
            seq(self.db)
                .map(lambda p: (p[0], self.flann.knnMatch(np.asarray(p[2], np.float32), np.asarray(des, np.float32), 2)))
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
            if m.distance < 0.7 * n.distance:
                good.append([m])
        return good

    def train(self, images: List[Picture]) -> None:
        for image in images:
            kp, des = self.sift.detectAndCompute(image.get_image(), None)
            self.db.append((image, kp, des))
