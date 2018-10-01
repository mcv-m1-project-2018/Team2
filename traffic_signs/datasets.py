from typing import List, Tuple
import numpy as np
import fnmatch
import os


class GroundTruth:
    top_left: Tuple[float]
    bottom_right: Tuple[float]
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

    def get_train_set(self):
        # TODO
        pass

    def get_validation_set(self):
        # TODO
        pass
