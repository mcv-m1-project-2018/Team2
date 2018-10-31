from typing import List, Tuple

from methods import AbstractMethod
from methods.operations import CompareHistograms, SIFTBrute
from model import Picture


class orb_brute(AbstractMethod):
    sift: SIFTBrute

    def __init__(self):
        self.orb = ORBBrute()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.orb.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.orb.train(images)


instance = orb_brute()
