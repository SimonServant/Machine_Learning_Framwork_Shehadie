from abc import ABC, abstractmethod
from typing import List, Tuple
from models.datatype import Tensor


class Layer(ABC):

    # Warum outTensors ? Warum Liste von Elementen ?
    @abstractmethod
    def forward(self, inTensor: Tensor, outTensor: Tensor):
        pass

    @abstractmethod
    def backward(self, outTensor: Tensor, inTensor: Tensor):
        pass

    # Falls die Parameter in den Layern sind
    @abstractmethod
    def calculateDeltaWeights(self, outTensor: Tensor, inTensor: Tensor):
        pass
