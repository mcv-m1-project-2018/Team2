import numpy as np


class Picture:
    image_cached: np.array
    name: str

    def __init__(self, name: str):
        self.name = name

    def get_image(self, parent: str) -> np.array:
        if self.image_cached is None:
            # TODO read image
            pass
        return self.image_cached
