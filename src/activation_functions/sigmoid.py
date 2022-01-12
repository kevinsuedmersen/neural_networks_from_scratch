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
    pass
