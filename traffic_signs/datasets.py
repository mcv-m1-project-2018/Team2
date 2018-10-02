import cv2
import fnmatch
import os
import random
from typing import Tuple, List

from functional import seq


class Rectangle:
    top_left: Tuple[float]
    width: float
    height: float

    def get_bottom_right(self):
        return self.top_left[0] + self.height, self.top_left[1] + self.width

    def get_area(self):
        return self.width * self.height


class GroundTruth:
    rectangle: Rectangle
    type: str


class Data:
    """Stores the content of a data element."""

    name: str
    gt: List[GroundTruth] = []
    img_path: str
    mask_path: str

    def __init__(self, directory: str, name: str):
        self.name = name
        self.img_path = '{}/{}'.format(directory, name)
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

        print(len(self.gt))

    def get_img(self):
        return cv2.imread(self.img_path)

    def get_mask_img(self):
        return cv2.imread(self.mask_path)


class DatasetManager:
    """We will use k-fold validation. More info at https://www.openml.org/a/estimation-procedures/1"""

    data: List[Data] = []
    _dir: str

    def __init__(self, directory: str):
        self._dir = directory

    def load_data(self):
        file_names = sorted(fnmatch.filter(os.listdir(self._dir), '*.jpg'))
        for file_name in file_names:
            self.data.append(Data(self._dir, file_name.replace('.jpg', '')))

    def get_data_by_type(self):
        # More info about how this works: https://github.com/EntilZha/PyFunctional#transformations-and-actions-apis
        types = seq(self.data).group_by(lambda sample: sample.gt[0].type).to_dict()
        return types

    def get_sets(self):
        """Get the validation and training sets of data from the original training dataset 30% to 70% from each class"""
        training = []
        verification = []
        types = self.get_data_by_type()
        for sign_type in ['A', 'B', 'C', 'D', 'E', 'F']:
            random.shuffle(types[sign_type])
            for j in types[sign_type]:
                if j <= 0.7*len(types[sign_type]):
                    training.append(types[sign_type][j])
                else:
                    verification.append(types[sign_type][j])

        return training, verification

    def get_sets_k_fold(self, k):
        # TODO
        pass


