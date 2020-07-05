import numpy as np


class Relu:

    def forward(self, x):
        result = np.maximum(x, 0)
        return result

    def backward(self, z):
        result = np.where(z > 0, 1.0, 0.0)
        return result
