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
    """Computes the Jacobian of the ReLU function. The ReLU takes a scalar as input and outputs a
    scalar as well, so the Jacobian of the ReLU function only has non-zero on its diagonal
    """
    batch_size = dendritic_potentials.shape[0]
    n_neurons = dendritic_potentials.shape[1]

    # Init the jacobians with zeros
    jacobians = np.zeros((batch_size, n_neurons, n_neurons))

    # Diagonal elements are 1 if dendritic_potential > 0 and 0 otherwise
    diagonal_elements = dendritic_potentials > 0  # (batch_size, n_neurons, 1)
    diagonal_elements = np.squeeze(diagonal_elements)  # (batch_size, n_neurons)
    diagonal_elements = diagonal_elements.astype(float)

    # Set diagonal elements into jacobians
    jacobians[:, np.arange(n_neurons), np.arange(n_neurons)] = diagonal_elements

    return jacobians

