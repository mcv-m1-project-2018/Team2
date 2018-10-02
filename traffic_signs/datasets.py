import fnmatch
import os
import random
from typing import Tuple, List

from functional import seq


class Size:
    height: float
    width: float

    def max(self, other):
        if self.height > other.height and self.width > other.width:
            return self
        elif self.height < other.height and self.width < other.width:
            return other
        else:
            size = Size()
            size.height = max(self.height, other.height)
            size.width = max(self.width, other.width)
            return size

    def min(self, other):
        if self.height < other.height and self.width < other.width:
            return self
        elif self.height > other.height and self.width > other.width:
            return other
        else:
            size = Size()
            size.height = min(self.height, other.height)
            size.width = min(self.width, other.width)
            return size


class GroundTruth:
    top_left: Tuple[float]
    bottom_right: Tuple[float]
    size: Size
    type: str


class Data:
    """Stores the content of a data element."""

    name: str
    gt: List[GroundTruth] = []
    img: str
    mask: str

    def __init__(self, directory: str, name: str):
        self.name = name
        self.img = '{}/{}'.format(directory, name)
        self.mask = '{}/mask/mask.{}.png'.format(directory, name)
        with open('{}/gt/gt.{}.txt'.format(directory, name)) as f:
            for line in f.readlines():
                parts = line.split(' ')
                gt = GroundTruth()
                gt.top_left = (parts[0], parts[1])
                gt.bottom_right = (parts[2], parts[3])
                gt.type = parts[4]
                gt.size.width = float(parts[3]) - float(parts[1])
                gt.size.height = float(parts[2]) - float(parts[0])
                self.gt.append(gt)


class DatasetManager:
    """We will use k-fold validation. More info at https://www.openml.org/a/estimation-procedures/1"""

    data: List[Data]
    _dir: str

    def __init__(self, directory: str):
        self._dir = directory

    def load_data(self):
        file_names = sorted(fnmatch.filter(os.listdir(self._dir), '*.jpg'))
        for file_name in file_names:
            self.data.append(Data(self._dir, file_name))
        print('Loading data')
    
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


