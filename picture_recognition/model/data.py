import fnmatch
import os
from typing import List
from model import Picture


class Data:
    data = List[Picture]
    _dir = str

    def __init__(self, directory: str):
        self.data = []
        self._dir = directory

        file_names = fnmatch.filter(os.listdir(self._dir), '*.jpg')
        for file_name in file_names:
            img_path = '{}/{}.jpg'.format(self._dir, file_name)
            self.data.append(Picture(img_path, file_name.replace('.jpg', '')))