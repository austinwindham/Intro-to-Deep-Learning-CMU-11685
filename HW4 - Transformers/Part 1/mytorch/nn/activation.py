import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        exp_Z = np.exp(Z - np.max(Z, axis= self.dim, keepdims=True))  

        self.A =  exp_Z / np.sum(exp_Z, axis= self.dim, keepdims=True) 
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            batch_size = np.prod(shape[:-1])
            A_reshaped = self.A.reshape(batch_size, C)
            dLdA_reshaped = dLdA.reshape(batch_size, C)
        else:
            A_reshaped = self.A
            dLdA_reshaped = dLdA

        
        # make jacobian
        A = A_reshaped[:, :, None] 
        AT = A_reshaped[:, None, :]
        I = np.eye(C)[None, :, :]

        dAdZ = A * I - A @ AT


        dLdZ = np.einsum("bij,bi->bj", dAdZ, dLdA_reshaped) # faster hing, same s doing the for loop

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            #self.A = NotImplementedError
            dLdZ = dLdZ.reshape(shape)

        return dLdZ
 

    