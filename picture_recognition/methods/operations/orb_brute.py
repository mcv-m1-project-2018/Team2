import math
import sys
from typing import List

import cv2
import numpy as np
from functional import seq
from tqdm import tqdm

from methods.operations.text import detect_text
from model import Picture, Frame
from model import Rectangle

THRESHOLD = 30


class ORBBrute:
    db: [(Picture, List[cv2.KeyPoint], np.array)]

    bf: cv2.BFMatcher
    orb: cv2.ORB

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(1000)

    def query(self, picture: Picture, frame: Frame = None) -> List[Picture]:
        if frame and frame.is_valid():
            im = picture.get_image()
            side = int(math.sqrt(frame.get_area()) * 0.8)
            m = frame.get_perspective_matrix(np.array([[0, side - 1], [side - 1, side - 1], [side - 1, 0], [0, 0]]))
            im = cv2.warpPerspective(im, m, (side, side))
        else:
            im = picture.get_image()

        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.show()

        kp, des = self.orb.detectAndCompute(im, None)

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
        for image in tqdm(images, total=len(images), file=sys.stdout, desc='Training orb'):
            mask, bounding_text = detect_text(image.get_image())
            if use_mask:
                kp, des = self.orb.detectAndCompute(image.get_image(), mask=mask)
            else:
                kp, des = self.orb.detectAndCompute(image.get_image(), None)

            self.db.append((image, kp, des))
            bounding_texts.append(bounding_text)
        return bounding_texts
