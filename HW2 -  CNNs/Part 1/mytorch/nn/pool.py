import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        
        self.max_indices = np.zeros((batch_size, in_channels, output_height, output_width), dtype=int)

        for i in range(output_height):
            for j in range(output_width):
                section = A[:, :, i: i+ self.kernel, j: j+ self.kernel]

                Z[:,:, i, j] = np.max(section, axis = (2,3))

                self.max_indices[:, :, i, j] = np.argmax(section.reshape(batch_size, in_channels, -1), axis=2)


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = output_height + self.kernel - 1
        input_width = output_width + self.kernel - 1

        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))

        for i in range(output_height):
            for j in range(output_width):

                fwd_max_idx = self.max_indices[:, :, i, j]

                max_row, max_col = np.unravel_index(fwd_max_idx, (self.kernel, self.kernel))

                for b in range(batch_size):
                    for c in range(in_channels):
                        dLdA[b, c, i + max_row[b, c], j + max_col[b, c]] += dLdZ[b, c, i, j]

        return dLdA






class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):

                section = A[:, :, i:i+self.kernel, j:j+self.kernel]

                Z[:, :, i, j] = np.mean(section, axis=(2, 3))

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = output_height + self.kernel - 1
        input_width = output_width + self.kernel - 1

        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))


        for i in range(output_height):
            for j in range(output_width):

                dLdA[:, :, i: i + self.kernel, j: j+ self.kernel] += (dLdZ[:, :, i, j])[:, :, None,None] / self.kernel**2

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_stride1 = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_stride1 = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z_stride1)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)

        return dLdA
