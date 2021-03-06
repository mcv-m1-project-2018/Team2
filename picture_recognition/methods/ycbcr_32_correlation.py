from typing import List

from methods import AbstractMethod
from methods.operations import CompareHistograms, CompareHistogramsMethods, HistogramTypes
from methods.operations.compare_histrograms import HistogramComparisonMethods
from model import Picture


class ycbcr_32_correlation(AbstractMethod):
    compare_histograms: CompareHistograms

    def __init__(self):
        self.compare_histograms = CompareHistograms(CompareHistogramsMethods.BLOCKS_32_32, HistogramTypes.YCbCr,
                                                    HistogramComparisonMethods.HISTCMP_CORREL)

    def query(self, picture: Picture) -> List[Picture]:
        res = self.compare_histograms.query(picture)
        return res

    def train(self, images: List[Picture]):
        self.compare_histograms.train(images)


instance = ycbcr_32_correlation()
