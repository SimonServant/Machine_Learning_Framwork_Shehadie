import numpy as np


class MeanSquaredErrorLoss:

    # loss function and its derivative
    def forward(self, y_actual, y_predicted):
        result = np.mean(np.power(y_actual - y_predicted, 2))
        return result

    def backward(self, y_actual, y_predicted):
        result = 2 * (y_predicted - y_actual) / y_actual.size
        return result
