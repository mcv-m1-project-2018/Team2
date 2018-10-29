from abc import abstractmethod, ABC
from typing import List, Tuple

from model import Picture


class AbstractMethod(ABC):

    @abstractmethod
    def query(self, picture: Picture) -> List[Picture]:
        pass

    @abstractmethod
    def train(self, images: List[Picture]):
        pass
