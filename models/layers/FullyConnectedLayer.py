from models.layers import Layer
from models.datatype import Shape, Tensor
import numpy as np


class FullyConnectedLayer(Layer):

    def __init__(self, InShape: Shape, OutShape: Shape):
        # Only for case of 2 D matrices, they have to fit in the correct spaces
        self.input_size = InShape[1]
        self.output_size = OutShape[0]
        # self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.weights = np.random.uniform(-1, 1, size=(self.input_size, self.output_size))
        self.bias = np.random.rand(self.output_size) - 0.5
        self.deltas = []
        self.deltas_bias = []

    def forward(self, inTensor: Tensor, outTensor: Tensor):
        outTensor.elements = np.dot(inTensor.elements, self.weights) + self.bias
        return

    def backward(self, outTensor: Tensor, inTensor: Tensor):
        deltaNow = np.dot(outTensor.deltas, self.weights.T)
        inTensor.setDelta(deltaNow)
        return

    def calculateDeltaWeights(self, outTensor: Tensor, inTensor: Tensor):
        self.deltas = np.dot(inTensor.elements.T, outTensor.deltas)
        self.deltas_bias = outTensor.deltas.sum(axis=0)
        pass
