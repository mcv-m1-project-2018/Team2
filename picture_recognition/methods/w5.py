from typing import List

from methods import AbstractMethod
from methods.operations import get_lines_rotation_and_crop
from model import Picture


class w5(AbstractMethod):

    def __init__(self):
        pass

    def query(self, picture: Picture) -> List[Picture]:
        get_lines_rotation_and_crop(picture.get_image())
        return []

    def train(self, images: List[Picture]):
        pass


instance = w5()
