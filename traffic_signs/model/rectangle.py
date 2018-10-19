from typing import Tuple, Optional


class Rectangle:
    """
    In the class rectangle we define a rectangle through the top left point, width and height
    
    In the function get_bottom_right with the top-left point and the values for width and height
    we find the bottom_right point
    
    Finally we compute the area of a rectangle in the function get_area 
    
    """

    top_left: (float, float)
    width: float
    height: float

    def __init__(self, top_left=(0, 0), width=0, height=0):
        self.top_left = top_left
        self.width = width
        self.height = height

    def get_bottom_right(self) -> (float, float):
        return self.top_left[0] + self.height, self.top_left[1] + self.width

    def get_bottom_left(self) -> (float, float):
        return self.top_left[0] + self.height, self.top_left[1]

    def get_top_right(self) -> (float, float):
        return self.top_left[0], self.top_left[1] + self.width

    def get_area(self) -> float:
        return self.width * self.height

    def get_form_factor(self) -> float:
        return self.width / self.height

    def clone(self) -> 'Rectangle':
        rec = Rectangle()
        rec.top_left = tuple(self.top_left)
        rec.width = self.width
        rec.height = self.height

        return rec

    def contains_point(self, point: (float, float)) -> bool:
        return (self.top_left[0] <= point[0] <= self.get_bottom_right()[0] and
                self.top_left[1] <= point[1] <= self.get_bottom_right()[1])

    def contains_rectangle(self, rectangle) -> bool:
        return self.contains_point(rectangle.top_left) or self.contains_point(rectangle.get_bottom_right()) or \
                self.contains_point(rectangle.get_top_right()) or self.contains_point(rectangle.get_bottom_left()) or \
                rectangle.contains_point(self.top_left) or rectangle.contains_point(self.get_bottom_right()) or \
                rectangle.contains_point(self.get_top_right()) or rectangle.contains_point(self.get_bottom_left())

    def union(self, other: 'Rectangle') -> 'Rectangle':
        rec = Rectangle()
        rec.top_left = (min(self.top_left[0], other.top_left[0]), min(self.top_left[1], other.top_left[1]))
        bottom_right = (min(self.get_bottom_right()[0], other.get_bottom_right()[0]),
                        min(self.get_bottom_right()[1], other.get_bottom_right()[1]))

        rec.width = (bottom_right[1] - self.top_left[1]) + 1
        rec.height = (bottom_right[0] - self.top_left[0]) + 1

        return rec

    def intersection(self, other: 'Rectangle') -> Optional['Rectangle']:
        rec = Rectangle()
        if self.contains_point(other.top_left):
            rec.top_left = other.top_left
            rec.height = (other.top_left[0] - self.get_bottom_right()[0]) + 1
            rec.width = (other.top_left[1] - self.get_bottom_right()[1]) + 1
        elif other.contains_point(self.top_left):
            rec.top_left = self.top_left
            rec.height = (self.top_left[0] - other.get_bottom_right()[0]) + 1
            rec.width = (self.top_left[1] - other.get_bottom_right()[1]) + 1
        else:
            return None

        return rec

    def __str__(self):
        return str(self.top_left) + ', ' + str(self.width) + 'x' + str(self.height)

    def to_csv(self):
        return str(self.top_left[0]) + ' ' + str(self.top_left[1]) + ' ' + str(self.get_bottom_right()[0]) + ' ' + \
               str(self.get_bottom_right()[1])
