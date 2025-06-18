import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:
    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use ReLU activations for the layer.
        """
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A0):
        """
        Create your own mytorch.models.MLP0!
        Pass the input through the linear layer followed by the activation layer to get the model output.
        Read the writeup (Hint: MLP0 Section) for further details on MLP0 forward and backward implementation.
        """
        Z0 = self.layers[0].forward(A0) # TODO
        A1 = self.layers[1].forward(Z0)  # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1  # TODO - What should be the return value?

    def backward(self, dLdA1):
        """
        Create your own mytorch.models.MLP0!
        Read the writeup (Hint: MLP0 Section) for further details on MLP0 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        dLdZ0 = self.layers[1].backward(dLdA1) # TODO
        dLdA0 = self.layers[0].backward(dLdZ0) # TODO

        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0 # TODO - What should be the return value?


class MLP1:
    def __init__(self, debug=False):
        """
        Initialize 2 linear layers.
        Layer 1 of shape (2,3).
        Layer 2 of shape (3, 2).
        Use ReLU activations for both the layers.
        Implement it on the same lines (in a list) as in the MLP0 class.
        """
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU(), Linear(3,2), ReLU()] # TODO

    def forward(self, A0):
        """
        Create your own mytorch.models.MLP1!
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        Read the writeup (Hint: MLP1 Section) for further details on MLP1 forward and backward implementation.
        """
        Z0 = self.layers[0].forward(A0) # TODO
        A1 = self.layers[1].forward(Z0) # TODO

        Z1 = self.layers[2].forward(A1) # TODO
        A2 = self.layers[3].forward(Z1) # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2 # TODO - What should be the return value?

    def backward(self, dLdA2):
        """
        Create your own mytorch.models.MLP1!
        Read the writeup (Hint: MLP1 Section) for further details on MLP1 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        dLdZ1 = self.layers[3].backward(dLdA2) # TODO
        dLdA1 = self.layers[2].backward(dLdZ1) # TODO

        dLdZ0 = self.layers[1].backward(dLdA1)  # TODO
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0 # TODO - What should be the return value?


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2).
        Note: Use ReLU activations for ALL the linear layers.
        Implement it on the same lines (in a list) as in the MLP1 class.

        Hint: Refer the diagrammatic view in the writeup for better understanding!
        """
        # Note: List of Hidden and Activation Layers in the correct order.
        self.debug = debug
        self.layers = [Linear(2,4), ReLU(), Linear(4,8), ReLU(), Linear(8,8), ReLU(), Linear(8,4), ReLU(), Linear(4,2), ReLU()] # TODO

    def forward(self, A):
        """
        Create your own mytorch.models.MLP4!
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        Read the writeup (Hint: MLP4 Section) for further details on MLP4 forward and backward implementation.
        """
        if self.debug:
            self.A = [A]

        L = len(self.layers)

        for i in range(L):
            A = self.layers[i].forward(A)  # TODO
            if self.debug:
                self.A.append(A)

        return A  # TODO - What should be the return value?

    def backward(self, dLdA):
        """
        Create your own mytorch.models.MLP4!
        Read the writeup (Hint: MLP4 Section) for further details on MLP4 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        if self.debug:
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):
            dLdA = self.layers[i].backward(dLdA)  # TODO
            if self.debug:
                self.dLdA = [dLdA] + self.dLdA

        return dLdA  # TODO - What should be the return value?
