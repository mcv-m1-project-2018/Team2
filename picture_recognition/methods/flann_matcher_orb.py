from typing import List

from methods import AbstractMethod
from methods.operations import Flann_Matcher_ORB
from model import Picture


class Flann_Matcher_ORB(AbstractMethod):
    flann_orb: Flann_Matcher_ORB

    def __init__(self):
        self.flann_orb = Flann_Matcher_ORB()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.flann_orb.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.flann_orb.train(images)


instance = Flann_Matcher_ORB()
