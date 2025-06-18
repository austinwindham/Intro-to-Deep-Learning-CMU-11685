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

from cnn import *
from activation import *
from loss import *

# Note: Feel free to change anything about this file for your testing purposes

np.random.seed(11785) # Set the seed so that the random values are the same each time


def sandbox_cnn_weight_init(out_channels, in_channels, kernel_size):
    return np.ones((out_channels, in_channels, kernel_size))

def sandbox_linear_weight_init(out_channels, in_channels):
    return np.ones((out_channels, in_channels))


def sandbox_bias_init(out_channels):
    return np.zeros(out_channels)


input_width = 8
num_input_channels = 2
num_channels = [4, 4, 4]
kernel_sizes = [2, 2, 2]

model = CNN(
    input_width = input_width,
    num_input_channels = num_input_channels,
    num_channels = num_channels,
    kernel_sizes = kernel_sizes,
    strides = [1, 1, 1],
    num_linear_neurons = 4,
    activations = [ReLU(), ReLU(), ReLU()],
    conv_weight_init_fn = sandbox_cnn_weight_init,
    bias_init_fn = sandbox_bias_init,
    criterion = CrossEntropyLoss(),
    linear_weight_init_fn = None,
    lr = 0.1
)

model.linear_layer.W = sandbox_linear_weight_init(model.linear_layer.W.shape[0], model.linear_layer.W.shape[1])

batch_size = 1
x = np.random.randn(batch_size, num_input_channels, input_width)

#TODO: Uncomment and/or add print statements as you need them.

# print("input shape: ", x.shape)
# print("input: ", x)

y = model.forward(x)

# print("Forward shape: ", y.shape)
# print("Forward result: ", y)

delta = np.ones(y.shape)

dx = model.backward(delta)

# print("Backward shape: ", dx.shape)
# print("Backward result: ", dx)
