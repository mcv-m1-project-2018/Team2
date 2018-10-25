import numpy as np
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


