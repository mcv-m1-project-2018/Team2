from typing import List

from model import Rectangle


def combine_overlapped_regions(regions: List[Rectangle]) -> List[Rectangle]:
    """
    Combines the possible overlapped regions into a set of non overlapped regions by using union.
    :param regions: the list of regions probably overlapped
    :return: a list of regions that are not overlapped
    """
    ret = []

    i = 0
    while i < len(regions):
        j = i + 1
        region = regions[i]
        while j < len(regions):
            if region.contains_rectangle(regions[j]):
                region = region.union(regions[j])
                regions.remove(j)
                j -= 1
            j += 1
        ret.append(region)
        i += 1

    return ret
