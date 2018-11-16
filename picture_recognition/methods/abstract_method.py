from abc import abstractmethod, ABC
from typing import List, Tuple

from model import Picture, Frame
from model.rectangle import Rectangle


class AbstractMethod(ABC):

    @abstractmethod
    def query(self, picture: Picture) -> (List[Picture], Frame):
        pass

    @abstractmethod
    def train(self, images: List[Picture]) -> List[Rectangle]:
        pass
