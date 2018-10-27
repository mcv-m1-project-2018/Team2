from typing import List, Tuple

from methods import AbstractMethod
from methods.operations import CompareHistograms, CompareHistogramsMethods, HistogramTypes
from methods.operations.compare_histrograms import HistogramComparisonMethods
from model import Picture


class hsv_16_hellinger(AbstractMethod):
    compare_histograms: CompareHistograms

    def __init__(self):
        self.compare_histograms = CompareHistograms(CompareHistogramsMethods.BLOCKS_16_16, HistogramTypes.HSV,
                                                    HistogramComparisonMethods.HISTCMP_HELLINGER)

    def query(self, picture: Picture) -> List[Tuple[Picture, float]]:
        res = self.compare_histograms.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.compare_histograms.train(images)


instance = hsv_16_hellinger()
