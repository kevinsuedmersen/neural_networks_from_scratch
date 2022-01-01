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


def softmax_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the backward pass of the softmax function.
    :param dendritic_potentials: shape = (batch_size, n_neurons_current_layer, 1)
    :param activations: shape = (batch_size, n_neurons_current_layer, 1)
    :return: shape = (batch_size, n_neurons_current_layer, n_neurons_current_layer)
    """
    raise NotImplementedError
