from typing import List
from methods import AbstractMethod
from methods.operations import SURFBrute
from model import Picture


class surf_brute(AbstractMethod):
    surf: SURFBrute

    def __init__(self):
        self.surf = SURFBrute()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.surf.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.surf.train(images)


instance = surf_brute()
