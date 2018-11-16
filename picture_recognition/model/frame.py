import math
from typing import List, Tuple
from functional import seq


class Frame:
    angle: float
    points: List[Tuple[int, int]]

    def __init__(self, points):
        self.points = self._add_points(points)
        self.angle = self._get_angle()

    def _add_points(self, not_sorted: List[Tuple[int, int]]):
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

    def _get_angle(self):
        angle = (math.atan2(self.points[0][1] - self.points[1][1], self.points[0][0] - self.points[1][0]))
        return math.degrees(angle) % 90

    def to_result(self):
        return [self.angle, self.points]
