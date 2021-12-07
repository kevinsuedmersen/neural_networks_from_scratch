from typing import Tuple

import numpy as np
import numpy.typing as npt

from src.types import BatchSize, NNeurons


def linear_forward(
        dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons]]
) -> npt.NDArray[Tuple[BatchSize, NNeurons]]:
    """Computes the forward pass of the linear activation function"""
    return dendritic_potentials


def relu_forward(
        dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons]]
) -> npt.NDArray[Tuple[BatchSize, NNeurons]]:
    """Computes the forward pass of the ReLU activation function"""
    activations = np.maximum(dendritic_potentials, 0)
    # TODO: Test that activations.shape == dendritic_potentials.shape

    return activations


def relu_backward():
    pass


def get_activation_function(activation_function_name: str):
    """Maps a string to an activation function"""
    if activation_function_name == "linear":
        return linear_forward

    if activation_function_name == "relu":
        return relu_forward

    elif activation_function_name == "sigmoid":
        raise NotImplementedError

    elif activation_function_name == "tanh":
        raise NotImplementedError

    elif activation_function_name == "softmax":
        raise NotImplementedError
