from typing import List, Tuple

from methods import AbstractMethod
from methods.operations import CompareHistograms, FLANN_Matcher
from model import Picture


class sift_brute(AbstractMethod):
    sift: FLANN_Matcher

    def __init__(self):
        self.orb = FLANN_Matcher()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.orb.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.orb.train(images)


instance = sift_brute()
