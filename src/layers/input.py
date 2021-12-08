from functools import reduce
from typing import Tuple

import numpy.typing as npt

from src.layers.interface import Layer


class InputLayer(Layer):
    def init_parameters(self, *args, **kwargs):
        pass

    def __init__(self, input_shape: Tuple):
        self.input_shape = input_shape
        # in self.forward(), all but the batch dimensions will be flattened. So, the size of the
        # flattened dimension will the product of all dimensions except for the batch dimension
        flattened_dim = reduce(lambda x, y: x*y, input_shape[1:])
        self.output_shape = (input_shape[0], flattened_dim)

    def forward(self, x_batch: npt.NDArray) -> npt.NDArray:
        """Makes sure x_batch is converted into an array of shape (batch_size, n_features)"""
        assert x_batch.shape[1:] == self.input_shape[1:], \
            f"Expected x_batch.shape={self.input_shape}, actual x_batch.shape={x_batch.shape}." \
            f"The input layer of the network needs consistent input shapes"
        x_batch_reshaped = x_batch.reshape((x_batch.shape[0], -1, 1))

        return x_batch_reshaped

    def backward(self):
        pass

    def compute_weight_grads(self):
        pass

    def compute_bias_grads(self):
        pass
