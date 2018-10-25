import numpy as np
import fnmatch
import os
from typing import List
import cv2


class Picture:
    image_cached: np.array
    name: str
    parent_dir: str

    def __init__(self, parent_dir: str, name: str):
        self.name = name
        self.parent_dir = parent_dir

    def get_image(self) -> np.array:
        if self.image_cached is None:
            self.image_cached = cv2.imread(self.parent_dir, cv2.IMREAD_COLOR)
        return self.image_cached


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
