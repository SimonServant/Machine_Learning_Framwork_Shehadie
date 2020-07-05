import numpy as np
from models.optimizers.flavours import VanillaStochasticGradientDescent
from models.activations import Sigmoid, Relu, Tanh
from models import NeuralNetwork

class SGDOptimizer():

    # flavour is method of parameter update
    def __init__(self, batchSize: int, learningRate: float, amountEpochs: int,  shuffle: bool = True, updateMechannism = VanillaStochasticGradientDescent):
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = shuffle
        self.updateMechanism = updateMechannism

    def optimizer(self, network: NeuralNetwork, data):
        pass