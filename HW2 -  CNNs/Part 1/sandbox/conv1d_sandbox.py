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
from Conv1d import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11485) # Set the seed so that the random values are the same each time

in_c = 2
out_c = 2
kernel = 2
width = 4
batch = 1

# initialize random input
x = np.array([[[1 , 2, 3, 4], [2, 2, 2,3]]])#np.random.randn(batch, in_c, width)
print ("input shape: ", x.shape)
print ("input: ", x)

print()

# weight init fn initializes a matrix of 0.5
def sandbox_weight_init_fn(out_channels, in_channels, kernel_size):
    return np.full((out_channels, in_channels, kernel_size), 0.5)

layer = Conv1d_stride1(
    in_c,
    out_c,
    kernel,
    sandbox_weight_init_fn,
    np.ones)

# TODO: Uncomment the following lines and change the file to test Conv1d
stride = 1
model = Conv1d(in_c, out_c, kernel, stride)

print(layer.b)
y = layer.forward(x)
#TODO: Uncomment and/or add print statements as you need them.

print("output shape: ", y.shape)
print("output: ", y)

delta = np.random.randn(*y.shape)
print()
print()
print()
print()

print("delta shape: ", delta.shape)
print("delta: ", delta)

dx = layer.backward(delta)

print("dx shape: ", dx.shape)
print("dx: ", dx)