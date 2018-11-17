from typing import List

from methods import AbstractMethod
from methods.operations import get_frame_with_lines, ORBBrute
from model import Picture, Frame
from model.rectangle import Rectangle


class w5(AbstractMethod):

    orb: ORBBrute

    def __init__(self):
        self.orb = ORBBrute()

    def query(self, picture: Picture) -> (List[Picture], Frame):
        frame = get_frame_with_lines(picture.get_image())

        return self.orb.query(picture), frame

    def train(self, images: List[Picture]) -> List[Rectangle]:
        # TODO get rectangles
        self.orb.train(images)
        return []


instance = w5()
