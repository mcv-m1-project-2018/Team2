import fnmatch
import os
import random
from typing import List

from functional import seq

from model import Data


class DatasetManager:
    """We will use k-fold validation. More info at https://www.openml.org/a/estimation-procedures/1"""

    data: List[Data]
    _dir: str

    def __init__(self, directory: str):
        self.data = []
        self._dir = directory

    def load_data(self):
        file_names = sorted(fnmatch.filter(os.listdir(self._dir), '*.jpg'))
        for file_name in file_names:
            self.data.append(Data(self._dir, file_name.replace('.jpg', '')))

    def get_data_by_type(self):
        # More info about how this works: https://github.com/EntilZha/PyFunctional#transformations-and-actions-apis
        types = seq(self.data).group_by(lambda sample: sample.gt[0].type).to_dict()
        return types

    def get_data_splits(self):
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

    def get_data_k_fold(self, k):
        # TODO
        pass

