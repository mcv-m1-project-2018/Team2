from typing import List, Tuple

from methods import AbstractMethod
from methods.operations import CompareHistograms, CompareHistogramsMethods, HistogramTypes
from methods.operations.compare_histrograms import HistogramComparisonMethods
from model import Picture, Frame, Rectangle


class ycbcr_16_hellinger(AbstractMethod):
    compare_histograms: CompareHistograms

    def __init__(self):
        self.compare_histograms = CompareHistograms(CompareHistogramsMethods.BLOCKS_16_16, HistogramTypes.YCbCr,
                                                    HistogramComparisonMethods.HISTCMP_HELLINGER)

    def query(self, picture: Picture) -> (List[Picture], Frame):
        res = self.compare_histograms.query(picture)
        return res, Frame()

    def train(self, images: List[Picture]) -> List[Rectangle]:
        self.compare_histograms.train(images)
        return [Rectangle() for _ in images]


instance = ycbcr_16_hellinger()
