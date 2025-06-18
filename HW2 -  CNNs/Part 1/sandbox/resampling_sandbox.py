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
from resampling import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11485) # Set the seed so that the random values are the same each time

# Create random input
in_c, out_c = 2, 2
batch, input_width = 1, 4
kernel = 2
x = np.random.randn(batch, in_c, input_width)

print("input shape: ", x.shape)
print("input: ", x)

# Initialize the resampling class
upsampling_factor = 2
upsample_1d = Upsample1d(upsampling_factor)

# TODO: Uncomment the following lines and change the file to test downsampling
#downsampling_factor = 2
#downsample_1d = Downsample1d(downsampling_factor)
########################################################################

# TODO: Uncomment the following lines and change the file to 2D versions
# x = np.random.randn(batch, in_c, input_width, input_width)
# upsample_2d = Upsample2d(upsampling_factor)
# downsample_2d = Downsample2d(downsampling_factor)
########################################################################

#TODO: Uncomment and/or add print statements as you need them.

# Perform forward and backward pass
forward_res = upsample_1d.forward(x)

# print("Forward shape: ", forward_res.shape)
# print("Forward result: ", forward_res)

backward_res = upsample_1d.backward(forward_res)

# print("Backward shape: ", backward_res.shape)
# print("Backward result: ", backward_res)