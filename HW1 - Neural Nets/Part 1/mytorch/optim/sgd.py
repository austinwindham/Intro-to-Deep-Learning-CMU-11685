import numpy as np


class SGD:
    """
    Create your own mytorch.optim.SGD!
    Read the writeup (Hint: Stochastic Gradient Descent (SGD) Section) for implementation details for the SGD class.
    Note: In scope of the homework, we are using linear layers.
    """

    def __init__(self, model, lr=0.1, momentum=0):
        # Initialize the model layers.
        self.l = model.layers
        self.L = len(model.layers)

        # Assign learning rate and momentum.
        self.lr = lr
        self.mu = momentum

        # Initialize momentum terms (velocity) for weights and biases to zeros.
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        # Update weights and biases for each layer.
        for i in range(self.L):
            if self.mu == 0:
                # If no momentum, use standard SGD update.
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW  # TODO: Update weights using gradients
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb  # TODO: Update biases using gradients
            else:
                # If momentum is used, update velocity terms.
                self.v_W[i] = self.mu * self.v_W[i] +  self.l[i].dLdW  # TODO: Update weight velocity
                self.v_b[i] = self.mu * self.v_b[i] +  self.l[i].dLdb  # TODO: Update bias velocity

                # Update weights and biases using momentum and learning rate.
                self.l[i].W = self.l[i].W - self.lr * self.v_W[i]  # TODO: Update weights using weight velocity
                self.l[i].b = self.l[i].b - self.lr * self.v_b[i]  # TODO: Update biases using bias velocity
