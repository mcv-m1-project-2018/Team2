from typing import List, Tuple
import cv2
from functional import seq
import numpy as np
from model import Picture
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4
THRESHOLD = 28


class ORBBruteHomography:
    db: List[Tuple[Picture, List[cv2.KeyPoint], np.array]]
    bf: cv2.BFMatcher
    orb: cv2.ORB

    def __init__(self):
        self.db = []
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(1000)

    def query(self, picture: Picture) -> List[Picture]:
        kp, des = self.orb.detectAndCompute(picture.get_image(), None)

        return (
            seq(self.db)
                .map(lambda p: (p[0], p[1], self.bf.match(des, p[2])))
                .map(lambda p: (p[0], p[1],
                                seq(p[2]).filter(lambda d: d.distance < max(THRESHOLD,
                                                                            seq(p[2]).map(lambda m: m.distance).min()))
                                .to_list()
                                )
                     )
                .filter(lambda p: len(p[2]) > MIN_MATCH_COUNT)
                .map(lambda p: (p[0], self._homography(kp, p[1], p[2], p[0].get_image(), picture.get_image())))
                .filter(lambda p: (p[1]) > MIN_MATCH_COUNT)
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
    def _homography(kp1, kp2, good, img2, img1):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches = mask.sum()

        matchesMask = mask.ravel().tolist()

        h, w, channel = np.asarray(img1).shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray'), plt.show()
        return matches

