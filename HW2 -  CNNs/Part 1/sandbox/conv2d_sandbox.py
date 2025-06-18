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
from Conv2d import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11685) # Set the seed so that the random values are the same each time

in_c = 2
out_c = 2
kernel = 2
width = 4
batch = 1

# initialize random input
x = np.random.randn(batch, in_c, width, width)
print("input shape: ", x.shape)
print("input: ", x)

# weight init fn initializes a matrix of 0.5
def sandbox_weight_init_fn(out_channels, in_channels, kernel_height, kernel_width):
    return np.full((out_channels, in_channels, kernel_height, kernel_width), 0.5)

conv_layer = Conv2d_stride1(
    in_c,
    out_c,
    kernel,
    sandbox_weight_init_fn,
    np.zeros)

# TODO: Uncomment the following lines and change the file to test Conv2d
# conv_layer = Conv2d(
#     in_channels=in_c, 
#     out_channels=out_c,
#     kernel_size=kernel,
#     stride=2,
#     padding=0,
#     weight_init_fn=sandbox_weight_init_fn, 
#     bias_init_fn=np.zeros)

y = conv_layer.forward(x)

#TODO: Uncomment and/or add print statements as you need them.

# print("output shape: ", y.shape)
# print("output: ", y)

delta = np.random.randn(*y.shape)

# print("delta shape: ", delta.shape)
# print("delta: ", delta)

dx = conv_layer.backward(delta)

# print("dx shape: ", dx.shape)
# print("dx: ", dx)