import numpy as np


class Tanh:

    def forward(self, x):
        result = np.tanh(x)
        return result

    def backward(self, z):
        result = 1 - np.tanh(z) ** 2
        return result
