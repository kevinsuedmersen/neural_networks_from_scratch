import numpy as np
import numpy.typing as npt


def relu_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """
    Computes the forward pass of the ReLU activation function
    :param dendritic_potentials: shape = (batch_size, n_neurons_current_layer, 1)
    :return: shape = (batch_size, n_neurons_current_layer, 1)
    """
    activations = np.maximum(dendritic_potentials, 0)

    return activations


def relu_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    raise NotImplementedError
