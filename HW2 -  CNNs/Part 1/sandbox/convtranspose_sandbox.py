import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Absolute path to the root directory
mytorch_dir = os.path.join(project_root, 'mytorch')
mytorch_nn_dir = os.path.join(mytorch_dir, 'nn')
models_dir = os.path.join(project_root, 'models')

sys.path.append(mytorch_dir)
sys.path.append(mytorch_nn_dir)
sys.path.append(models_dir)

from flatten import *
from ConvTranspose import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11785) # Set the seed so that the random values are the same each time

in_c, out_c = 2, 2
batch, width = 1, 4
kernel, upsampling_factor = 2, 2

# initialize random input
x = np.random.randn(batch, in_c, width)


# weight fn initializes a matrix of 1s
def sandbox_weight_init_fn(out_channels, in_channels, kernel_size):
    return np.ones((out_channels, in_channels, kernel_size))

# bias fn initializes a matrix of 0s
def sandbox_bias_init_fn(out_channels):
    return np.zeros(out_channels)

# layer declaration
conv_transpose_1d = ConvTranspose1d(
    in_channels=in_c, 
    out_channels=out_c,
    kernel_size=kernel,
    upsampling_factor=upsampling_factor,
    weight_init_fn=sandbox_weight_init_fn,
    bias_init_fn=sandbox_bias_init_fn)


#Test outputs
#TODO: Uncomment and/or add print statements as you need them.

y = conv_transpose_1d.forward(x)
# print("output shape: ", y.shape)
# print("output: ", y)

delta = np.ones(y.shape)
# print("delta shape: ", delta.shape)
# print("delta: ", delta)

conv_backward_res = conv_transpose_1d.backward(delta)
# print("dx shape: ", conv_backward_res.shape)
# print("dx: ", conv_backward_res)
