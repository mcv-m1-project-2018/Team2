from typing import List, Tuple

from methods import AbstractMethod
from methods.operations import CompareHistograms, CompareHistogramsMethods, HistogramTypes
from methods.operations.compare_histrograms import HistogramComparisonMethods
from model import Picture


class ycbcr_16_hellington(AbstractMethod):
    compare_histograms: CompareHistograms

    def __init__(self):
        self.compare_histograms = CompareHistograms(CompareHistogramsMethods.BLOCKS_16_16, HistogramTypes.YCbCr,
                                                    HistogramComparisonMethods.HISTCMP_HELLINGER)

    def query(self, picture: Picture) -> List[Tuple[Picture, float]]:
        res = self.compare_histograms.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.compare_histograms.train(images)


instance = ycbcr_16_hellington()
