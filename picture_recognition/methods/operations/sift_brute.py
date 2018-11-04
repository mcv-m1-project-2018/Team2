from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture

THRESHOLD = 28


class SIFTBrute:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array]]
    bf: cv2.BFMatcher
    sift: cv2.xfeatures2d.SIFT_create

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
        self.sift = cv2.xfeatures2d.SIFT_create(500)

    def query(self, picture: Picture) -> List[Picture]:
        kp, des = self.sift.detectAndCompute(picture.get_image(), None)

        return (
            seq(self.db)
                .map(lambda p: (p[0], self.bf.match(p[2], des)))
                .map(lambda p: (p[0],
                                seq(p[1]).filter(lambda d: d.distance < max(THRESHOLD,
                                                                            seq(p[1]).map(lambda m: m.distance).min()))
                                .to_list()
                                )
                     )
                .map(lambda p: (p[0], len(p[1])))
                .filter(lambda p: p[1] > 0)
                .sorted(lambda p: p[1], reverse=True)
                .map(lambda p: p[0])
                .take(10)
                .to_list()
        )

    def train(self, images: List[Picture]) -> None:
        for image in images:
            kp, des = self.sift.detectAndCompute(image.get_image(), None)
            self.db.append((image, kp, des))
