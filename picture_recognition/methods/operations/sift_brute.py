from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture
from .text import detect_text
from model import Rectangle

THRESHOLD = 27


class SIFTBrute:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array,Rectangle]]
    bf: cv2.BFMatcher
    sift: cv2.xfeatures2d.SIFT_create

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
        self.sift = cv2.xfeatures2d.SIFT_create(1000)

    def query(self, picture: Picture) -> List[Picture]:
        mask ,rec = detect_text(picture.get_image())
        kp, des = self.sift.detectAndCompute(picture.get_image(), mask)

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
                .filter(lambda p: p[1] > 4)
                .sorted(lambda p: p[1], reverse=True)
                .map(lambda p: p[0])
                .take(10)
                .to_list()
        )

    def train(self, images: List[Picture]) -> None:
        for image in images:
            mask,bounding_text = detect_text(image.get_image())
            kp, des = self.sift.detectAndCompute(image.get_image(), mask)
            self.db.append((image, kp, des,bounding_text))
