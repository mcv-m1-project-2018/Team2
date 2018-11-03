from typing import List
from methods import AbstractMethod
from methods.operations import BRIEF
from model import Picture


class brief(AbstractMethod):
    brief: BRIEF

    def __init__(self):
        self.brief = BRIEF()

    def query(self, picture: Picture) -> List[Picture]:
        res = self.brief.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.brief.train(images)


instance = brief()