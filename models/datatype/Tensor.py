from models.datatype import Shape


class Tensor:
    def __init__(self, elements, shape: Shape = None):
        self.elements = elements
        # only useful for different languages. Python is able to already implicitly cast and reshape
        # arrays to matrices and vice versa
        self.shape = shape
        self.deltas = []

    def setDeltas(self, deltas):
        self.deltas = deltas
