import numpy as np
import numpy.typing as npt


def sigmoid_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Implements the forward pass of the sigmoid function
    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, 1)
    """
    activations = 1 / (1 + np.exp(-dendritic_potentials))

    return activations


def sigmoid_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the Jacobians of the dendritic potentials. Since the sigmoid function has a scalar
    input and a scalar output, the sigmoid's Jacobians have zeros off the main diagonal. The diagonal
    elements equal a * (1 - a).
    :param dendritic_potentials: shape(batch_size, n_neurons_current_layer, 1)
    :param activations: shape(batch_size, n_neurons_current_layer, 1)
    :return: shape(batch_size, n_neurons_current_layer, n_neurons_current_layer))
    """
    # Get dimensions
    batch_size = activations.shape[0]
    n_neurons = activations.shape[1]

    # Init jacobians with zeros
    jacobians = np.zeros((batch_size, n_neurons, n_neurons))

    # Replace diagonal elements
    diagonal_elements = np.squeeze(activations * (1 - activations))
    jacobians[:, np.arange(n_neurons), np.arange(n_neurons)] = diagonal_elements

    return jacobians
