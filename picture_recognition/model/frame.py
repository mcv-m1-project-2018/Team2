import math
from typing import List

import cv2
import numpy as np
from functional import seq


class Frame:
    angle: float
    points: [(int, int), (int, int), (int, int), (int, int)]

    def __init__(self, points: [(int, int), (int, int), (int, int), (int, int)], angle: float):
        self.points = self._sort_points(points)
        self.angle = angle

    @staticmethod
    def _sort_points(not_sorted: [(int, int), (int, int), (int, int), (int, int)]):
        angles = List[float]
        center_x = (seq(not_sorted)
                    .map(lambda p: not_sorted[0])
                    .average())
        center_y = (seq(not_sorted)
                    .map(lambda p: not_sorted[1])
                    .average())
        for t in not_sorted:
            angles.append(math.atan2(t[1] - center_y, t[0] - center_x))
        sorted_idx = sorted(range(len(angles)), key=lambda x: angles[x], reverse=True)
        points = [not_sorted[i] for i in sorted_idx]
        return points

    def get_perspective_matrix(self, dst: np.array):
        return cv2.getPerspectiveTransform(self.points, dst)

    def to_result(self):
        return [self.angle, self.points]
