import numpy as np
import numpy.typing as npt


def linear_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Computes the forward pass of the linear activation function
    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, 1)
    """
    return dendritic_potentials


def linear_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the Jacobian of the activations in layer l w.r.t. the dendritic potentials in layer
    l for the linear activation function. The Jacobian of the linear activation function is the
    identity function, because its derivative is always 1 (constant) and because it maps a
    scalar input to a scalar output.
    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :param activations: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, n_neurons_current_layer)
    """
    jacobian = np.eye(dendritic_potentials.shape[1])  # for 1 element in the batch
    jacobian = jacobian[np.newaxis, ...]  # add axis to be repeated
    jacobians = np.repeat(jacobian, repeats=dendritic_potentials.shape[0], axis=0)  # batch_size identity matrices

    return jacobians
