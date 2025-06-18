# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size )) # TODO
        
        for i in range(output_size):
            Z[:, :, i] = np.tensordot(self.A[:, :, i: i + self.kernel_size] , self.W, axes = ([1, 2], [1, 2]))

        Z += self.b.reshape(1, self.out_channels, 1) 

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = np.zeros(self.W.shape)  # TODO

        batch_size, out_channels, output_size = dLdZ.shape

        self.dLdb = np.sum(dLdZ, axis = (0,2)) # i htink thi sis good


        # Create slices of A to match dLdZ
        for i in range(self.kernel_size):
            self.dLdW[:,:, i] = np.tensordot(dLdZ, self.A[:, :, i: i + output_size], axes=([0, 2], [0, 2]))

        # Compute dLdW using tensordot

        pad_size = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad_size, pad_size)), mode='constant')


        W_flip = np.flip(self.W, axis=2)

        dLdA = np.zeros_like(self.A)


        for i in range(self.A.shape[2]):
            
            dLdA[:, :, i] = np.tensordot(dLdZ_padded[:,:, i:i+self.kernel_size], W_flip, axes=([1, 2], [0, 2]))



        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')


        # Call Conv1d_stride1
        # TODO
        Z_stride1 = self.conv1d_stride1.forward(A_padded)


        # downsample
        Z = self.downsample1d.forward(Z_stride1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)


        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_upsampled)  # TODO

        # Unpad the gradient
        # TODO
        if self.pad == 0:
            return dLdA
        else:
            dLdA = dLdA[:, :, self.pad: -self.pad]


        return dLdA
