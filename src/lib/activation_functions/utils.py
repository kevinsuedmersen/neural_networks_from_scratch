from typing import Tuple

import numpy as np
import numpy.typing as npt


def init_sigmoid_tanh_forward(dendritic_potentials: npt.NDArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to initialize the forward pass of the sigmoid or tanh activation functions"""
    # Get the indices of the positive/negative inputs
    positive = (dendritic_potentials >= 0)
    negative = (dendritic_potentials < 0)

    # Init activations
    activations = np.zeros(dendritic_potentials.shape)

    return positive, negative, activations


def init_sigmoid_tanh_backward(activations: npt.NDArray) -> Tuple[int, int, np.ndarray]:
    """Helper function to initialize the backward pass of the sigmoid or tanh activation functions"""
    # Get dimensions
    batch_size = activations.shape[0]
    n_neurons = activations.shape[1]

    # Init jacobians with zeros
    jacobians = np.zeros((batch_size, n_neurons, n_neurons))

    return batch_size, n_neurons, jacobians
