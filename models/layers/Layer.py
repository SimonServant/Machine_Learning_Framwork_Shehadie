from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

# Falls die Parameter in den Layern sind
    @abstractmethod
    def calculateDeltaWeights(self):
        pass