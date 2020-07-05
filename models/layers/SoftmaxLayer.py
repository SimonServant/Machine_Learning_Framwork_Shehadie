import numpy as np
from models.layers import Layer
from models.datatype import Shape, Tensor

class SoftmaxLayer(Layer):
    
    def __init__(self):
        super().__init__()


    # Change to fit. There is never multiple Tensors in neural network. Parallelism can be reached otherwise
    def forward(self, inTensor: Tensor, outTensor: Tensor):
        x = inTensor.elements
        e_x = np.exp(x - np.max(x))
        sum_over_every_row = e_x.sum(axis=1, keepdims=True)
        result = e_x / sum_over_every_row
        outTensor.elements = result
        return

    def backward(self,outTensor: Tensor, inTensor: Tensor ):
        # in Combination with the Cross Entropy we can simple pass the gradient or in this case set the gradient to one
        inTensor.setDeltas(1)
        return
        

    # is not used
    def calculateDeltaWeights(self, outTensor: Tensor, inTensor: Tensor):
        raise NotImplementedError("The method calculate Delta weights is never used and therefore not implemented in softmax layer")