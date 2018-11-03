from typing import List

from methods import AbstractMethod
from methods.operations import SIFTBruteRatioTest
from model import Picture


class sift_brute_ratio_test(AbstractMethod):
    sift: SIFTBruteRatioTest

    def __init__(self):
        self.sift = SIFTBruteRatioTest()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.sift.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.sift.train(images)


instance = sift_brute_ratio_test()
