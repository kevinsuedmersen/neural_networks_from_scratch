from typing import Union, Tuple

import numpy as np
import numpy.typing as npt


def softmax_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """
    Computes the forward pass of the softmax activation function
    :param dendritic_potentials: shape = (batch_size, n_neurons_current_layer, 1)
    :return: shape = (batch_size, n_neurons_current_layer, 1)
    """
    # Subtract max (or any other constant) for numerical stability. It will cancel out,
    # because the max is also used in the denominator, i.e. exp_sum
    exp = np.exp(dendritic_potentials - np.max(dendritic_potentials))
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    activations = exp / exp_sum

    return activations


def softmax_backward(
        dendritic_potentials: npt.NDArray,
        activations: npt.NDArray,
        debug: bool = False
) -> Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray]:
    """Computes the jacobian of the softmax activation function
    :param dendritic_potentials: (batch_size, n_neurons_current_layer, 1)
    :param activations: (batch_size, n_neurons_current_layer, 1)
    :param debug: If True, also the diagonal elements of the Jacobian are returned
    :return: (batch_size, n_neurons_current_layer, n_neurons_current_layer)
    """
    # Take the (negative) outer matmul product from the activations
    # (batch_size, 1, n_neurons)
    activations_t = np.transpose(activations, axes=[0, 2, 1])
    # (batch_size, n_neurons, n_neurons)
    jacobians = np.matmul(activations, activations_t)
    jacobians *= (-1)

    # Replace the entries along the diagonal of ``jacobians`` for all batches
    n_neurons = activations.shape[1]
    diagonal_elements = activations * (1 - activations)
    diagonal_elements = np.squeeze(diagonal_elements)
    jacobians[:, np.arange(n_neurons), np.arange(n_neurons)] = diagonal_elements

    # TODO: In the tests, retrieve diagonal elements and remove debug flag here
    if debug:
        return jacobians, diagonal_elements
    else:
        return jacobians
