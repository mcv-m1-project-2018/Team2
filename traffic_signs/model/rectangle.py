from typing import Tuple


class Rectangle:
    top_left: Tuple[float]
    width: float
    height: float

    def get_bottom_right(self):
        return self.top_left[0] + self.height, self.top_left[1] + self.width

    def get_area(self):
        return self.width * self.height

    def __str__(self):
        return str(self.top_left) + ', ' + str(self.width) + 'x' + str(self.height)
