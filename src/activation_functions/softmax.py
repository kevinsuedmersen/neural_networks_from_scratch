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


def softmax_jacobian(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the jacobian of the softmax activation function
    :param dendritic_potentials: (batch_size, n_neurons_current_layer, 1)
    :param activations: (batch_size, n_neurons_current_layer, 1)
    :return: (batch_size, n_neurons_current_layer, n_neurons_current_layer)
    """
    # Take the (negative) outer matmul product from the activations
    # TODO: Only transpose the last 2 dimensions
    outer_product = np.matmul(activations, activations.T)  # (batch_size, n_neurons, n_neurons)
    outer_product *= (-1)

    # Replace the entries along the diagonal for all batches
    diagonal_elements = activations * (1 - activations)
    n_neurons = activations.shape[1]
    diagonal_idxs = (np.arange(n_neurons), np.arange(n_neurons))
    outer_product[:, diagonal_idxs] = diagonal_elements

    return outer_product


def softmax_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the backward pass of the softmax function.
    :param dendritic_potentials: shape = (batch_size, n_neurons_current_layer, 1)
    :param activations: shape = (batch_size, n_neurons_current_layer, 1)
    :return: shape = (batch_size, n_neurons_current_layer, n_neurons_current_layer)
    """
    raise NotImplementedError
