from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture


THRESHOLD=28

class BRIEF:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array]]
    bf: cv2.BFMatcher
    star: cv2.xfeatures2d_StarDetector
    brief:cv2.xfeatures2d_BriefDescriptorExtractor

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief= cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def query(self, picture: Picture) -> List[Picture]:
        kp = self.star.detect(picture.get_image(), None)
        kp, des = self.brief.compute(picture.get_image(), kp)

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
            kp = self.star.detect(image.get_image(), None)
            kp, des = self.brief.compute(image.get_image(), kp)
            self.db.append((image, kp, des))
