import numpy as np
import numpy.typing as npt


def tanh_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Implements the forward pass of the tanh function
    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, 1)
    """
    activations = (
        (np.exp(dendritic_potentials) - np.exp(-dendritic_potentials)) /
        (np.exp(dendritic_potentials) + np.exp(-dendritic_potentials))
    )

    return activations


def tanh_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the Jacobians of the tanh function w.r.t. the dendritic potentials. Since the tanh
    function has a scalar input and a scalar output, the tanh's Jacobians have zeros off the main
    diagonal. The diagonal elements equal 1 - a^2.
    :param dendritic_potentials: shape(batch_size, n_neurons_current_layer, 1)
    :param activations: shape(batch_size, n_neurons_current_layer, 1)
    :return: shape(batch_size, n_neurons_current_layer, n_neurons_current_layer))
    """
    # Get dimensionseeew
    batch_size = activations.shape[0]
    n_neurons = activations.shape[1]

    # Init jacobians with zeros
    jacobians = np.zeros((batch_size, n_neurons, n_neurons))

    # Replace diagonal elements
    diagonal_elements = np.squeeze(1 - activations**2)
    jacobians[:, np.arange(n_neurons), np.arange(n_neurons)] = diagonal_elements

    return jacobians
