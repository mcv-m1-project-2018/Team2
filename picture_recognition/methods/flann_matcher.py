from typing import List

from methods import AbstractMethod
from methods.operations import FLANN_Matcher
from model import Picture


class flann_matcher(AbstractMethod):
    flann: FLANN_Matcher

    def __init__(self):
        self.flann = FLANN_Matcher()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.flann.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.flann.train(images)


instance = flann_matcher()
