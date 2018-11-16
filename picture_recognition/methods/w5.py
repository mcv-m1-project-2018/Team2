from typing import List

from methods import AbstractMethod
from methods.operations import get_lines_rotation_and_crop, ORBBrute
from model import Picture, Frame
from model.rectangle import Rectangle


class w5(AbstractMethod):

    orb: ORBBrute

    def __init__(self):
        self.orb = ORBBrute()

    def query(self, picture: Picture) -> (List[Picture], Frame):
        points, angle = get_lines_rotation_and_crop(picture.get_image())

        frame = Frame(points, angle)
        return self.orb.query(picture), frame

    def train(self, images: List[Picture]) -> List[Rectangle]:
        # TODO get rectangles
        self.orb.train(images)


instance = w5()
