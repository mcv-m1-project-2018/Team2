import cv2
from typing import List

from model import GroundTruth, Rectangle


class Data:
    """Stores the content of a data element."""

    name: str
    gt: List[GroundTruth]
    img_path: str
    mask_path: str

    def __init__(self, directory: str, name: str):
        self.name = name
        self.gt = []
        self.img_path = '{}/{}.jpg'.format(directory, name)
        self.mask_path = '{}/mask/mask.{}.png'.format(directory, name)
        with open('{}/gt/gt.{}.txt'.format(directory, name)) as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                gt = GroundTruth()
                gt.type = parts[4]
                gt.rectangle = Rectangle()
                gt.rectangle.top_left = (float(parts[0]), float(parts[1]))
                gt.rectangle.width = float(parts[3]) - float(parts[1]) + 1
                gt.rectangle.height = float(parts[2]) - float(parts[0]) + 1
                self.gt.append(gt)

    def get_img(self):
        return cv2.imread(self.img_path, cv2.IMREAD_COLOR)

    def get_mask_img(self):
        return cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)