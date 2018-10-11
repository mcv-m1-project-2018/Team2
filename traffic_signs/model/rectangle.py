from typing import Tuple


class Rectangle:
    """
    In the class rectangle we define a rectangle through the top left point, width and height
    
    In the function get_bottom_right with the top-left point and the values for width and height
    we find the bottom_right point
    
    Finally we compute the area of a rectangle in the function get_area 
    
    """

    top_left: Tuple[float]
    width: float
    height: float

    def __init__(self, top_left=(0, 0), width=0, height=0):
        self.top_left = top_left
        self.width = width
        self.height = height

    def get_bottom_right(self):
        return self.top_left[0] + self.height, self.top_left[1] + self.width

    def get_area(self):
        return self.width * self.height

    def get_form_factor(self):
        return self.width / self.height

    def __str__(self):
        return str(self.top_left) + ', ' + str(self.width) + 'x' + str(self.height)
