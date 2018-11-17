from typing import List, Tuple

import cv2
from functional import seq
import numpy as np
from model import Picture, Frame
from methods.operations.text import detect_text
from model import Rectangle

THRESHOLD = 28


class ORBBrute:
    db: [(Picture, List[cv2.KeyPoint], np.array)]

    bf: cv2.BFMatcher
    orb: cv2.ORB

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(1000)

    def query(self, picture: Picture, frame: Frame = None) -> List[Picture]:
        if frame:
            # TODO frame transform
            im = picture.get_image()
        else:
            im = picture.get_image()
        mask, rec = detect_text(im)
        kp, des = self.orb.detectAndCompute(im, mask)

        return (
            seq(self.db)
                .map(lambda p: (p[0], self.bf.match(p[2], des)))
                .map(lambda p: (p[0],
                                seq(p[1])
                                .filter(lambda d: d.distance < max(THRESHOLD,
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

    def train(self, images: List[Picture], use_mask=True) -> List[Rectangle]:
        bounding_texts = []
        for image in images:
            mask, bounding_text = detect_text(image.get_image())
            if use_mask:
                kp, des = self.orb.detectAndCompute(image.get_image(), mask=mask)
            else:
                kp, des = self.orb.detectAndCompute(image.get_image(), None)

            self.db.append((image, kp, des))
            bounding_texts.append(bounding_text)
        return bounding_texts
