import numpy as np


class CrossEntropyLoss:

    def forward(self, y_actual, y_predicted):
        # we need to clip because `h` must never be <= 0, which
        # doesn't happen in theory, but will happen in praxis because of
        # automatic rounding
        y_predicted = np.clip(y_predicted, a_min=0.000000001, a_max=None)
        # - np.sum(y_one_hot * np.log(h_one_hot))

        N = y_predicted.shape[0]
        ce = -np.sum(y_actual * np.log(y_predicted)) / N
        return ce

    def backward(self, y_actual, y_predicted):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector.
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
                m = y.shape[0]
        grad = X
        grad[range(m), y] -= 1
        grad = grad/m
        return grad
        """
        return y_predicted - y_actual
