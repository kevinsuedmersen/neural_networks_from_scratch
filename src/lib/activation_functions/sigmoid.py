import numpy as np
import numpy.typing as npt


def sigmoid_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Implements the forward pass of the sigmoid function. For positive inputs, we will define the
    sigmoid function as usual, i.e. `f(z) = 1 / (1 + e^(-z))` and for negative inputs, we will reformulate
    the sigmoid function as `f(z) = e^z / (1 + e^z)`. The first version can handle large positive
    inputs and the second version can handle large negativ inputs. Together these 2 versions
    provide more numerical stability.

    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, 1)
    """
    # Get the indices of the positive/negative inputs
    positive = dendritic_potentials >= 0
    negative = dendritic_potentials < 0

    # Init activations
    activations = np.zeros(dendritic_potentials.shape)

    # For positive inputs, we use the usual version of the sigmoid function
    activations[positive] = 1 / (1 + np.exp(-dendritic_potentials[positive]))

    # For negative inputs, we use a slightly different version
    activations[negative] = (
            np.exp(dendritic_potentials[negative]) /
            (1 + np.exp(dendritic_potentials[negative]))
    )

    return activations


def sigmoid_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    """Computes the Jacobians of the sigmoid function w.r.t. the dendritic potentials.
    Since the sigmoid function has a scalar input and a scalar output, the sigmoid's Jacobians have
    zeros off the main diagonal. The diagonal elements equal a * (1 - a).
    :param dendritic_potentials: shape=(batch_size, n_neurons_current_layer, 1)
    :param activations: shape=(batch_size, n_neurons_current_layer, 1)
    :return: shape=(batch_size, n_neurons_current_layer, n_neurons_current_layer))
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
