from typing import List

from methods import AbstractMethod
from methods.operations import ORBBruteRatioTestHomography
from model import Picture


class orb_brute_ratio_test_homography(AbstractMethod):
    orb: ORBBruteRatioTestHomography

    def __init__(self):
        self.orb = ORBBruteRatioTestHomography()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.orb.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.orb.train(images)


instance = orb_brute_ratio_test_homography()
