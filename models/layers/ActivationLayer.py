import numpy as np

from models.layers import Layer
from models.datatype import Shape, Tensor
from models.activations import Sigmoid


class ActivationLayer(Layer):

    def __init__(self, activation=Sigmoid):
        super().__init__()
        self.activation = activation

    def forward(self, inTensor: Tensor, outTensor: Tensor):
        outTensor.elements = self.activation.forward(inTensor.elements)
        return

    def backward(self, outTensor: Tensor, inTensor: Tensor):
        result = np.dot(self.activation.backward(inTensor.elements), outTensor.elements)
        inTensor.setDeltas(result)
        return

    def calculateDeltaWeights(self, outTensor: Tensor, inTensor: Tensor):
        raise NotImplementedError()
