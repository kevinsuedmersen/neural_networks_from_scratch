import logging
from functools import reduce
from typing import Tuple

import numpy.typing as npt

from src.lib.layers import Layer

logger = logging.getLogger(__name__)


class InputLayer(Layer):
    def __init__(self, input_shape: Tuple, layer_idx: int = None):
        super().__init__()
        self.input_shape = input_shape
        self.layer_idx = layer_idx

        # in self.forward(), all but the batch dimensions will be flattened. So, the size of the
        # flattened dimension will the product of all dimensions except for the batch dimension
        flattened_dim = reduce(lambda x, y: x*y, input_shape[1:])
        self.output_shape = (input_shape[0], flattened_dim)
        self.activations = None
        self.weights = None
        self.biases = None
        self.weight_gradients = None
        self.bias_gradients = None

    def init_parameters(self, *args, **kwargs):
        """This method is called in the model.add_layer method and hence, it must be implemented here"""
        logger.info("Skipping parameter initialization of the input layer, because it has no parameters")

    def forward_propagate(self, x_batch: npt.NDArray) -> npt.NDArray:
        """Makes sure x_batch is converted into an array of shape (batch_size, n_features)"""
        assert x_batch.shape[1:] == self.input_shape[1:], \
            f"Expected x_batch.shape={self.input_shape}, actual x_batch.shape={x_batch.shape}." \
            f"The input layer of the network needs consistent input shapes"
        self.activations = x_batch.reshape((x_batch.shape[0], -1, 1))

        return self.activations

    def backward_propagate(self):
        raise NotImplementedError(
            "The input layer has no parameters and hence no errors should be computed during "
            "backpropagation"
        )

    def compute_weight_gradients(self):
        raise NotImplementedError(
            "The input layer has no parameters, so no weight gradients should be computed during "
            "backpropagation"
        )

    def compute_bias_gradients(self):
        raise NotImplementedError(
            "The input layer has no parameters, so no bias gradients should be computed during "
            "backpropagation"
        )
