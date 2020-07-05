import numpy as np


class Sigmoid:

    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    def backward(self, z):
        result = self.forward(z) * (1 - self.forward(z))
        return result
