import numpy as np
from typing import List, Tuple
from models.layers import Layer, InputLayer


class NeuralNetwork:
    # optional: Use parameters / weights as additional List here.
    def __init__(self, input: InputLayer, layers: List[Layer], loss):
        self.inputLayer = input
        self.layers = layers
        self.loss = loss
        # only needed for parallelization
        self.caches = []
        self.deltaParams = []

    # result of forward pass is loss
    def forward(self):
        output = self.layers[0].forward(self.inputLayer, )
        # for layer in self.layers:
        # TODO

        pass

    # backward pass uses loss derivative and initializes first delta. then gives this delta to backprop
    # data is loss derivative
    def backprop(self, data):
        pass
