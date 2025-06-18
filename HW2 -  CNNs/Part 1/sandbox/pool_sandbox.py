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
from pool import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11685) # Set the seed so that the random values are the same each time

kernel = 2
width = 4
in_c = 2
batch = 1

# Create sample input
x = np.random.randn(batch, in_c, width, width)

print("input shape: ", x.shape)
print("input: ", x)

# Initialize the MeanPool2d_stride1 class
pool_layer = MeanPool2d_stride1(kernel)

# TODO: Uncomment this line and change file to test MaxPool2d_stride1
# pool_layer = MaxPool2d_stride1(kernel)

# TODO: Uncomment these lines and change file to test MaxPool2d or MeanPool2d
# stride = 3
# pool_layer = MeanPool2d(kernel, stride)
# pool_layer = MaxPool2d(kernel, stride)

#TODO: Uncomment and/or add print statements as you need them.

forward_res = pool_layer.forward(x)

print("Forward Shape: ", forward_res.shape)
print("Forward Result: ", forward_res)

backward_res = pool_layer.backward(forward_res)

print("Backward Shape: ", backward_res.shape)
print("Backward Result: ", backward_res)
