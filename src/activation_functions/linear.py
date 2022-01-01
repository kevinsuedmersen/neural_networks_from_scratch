import numpy.typing as npt


def linear_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Computes the forward pass of the linear activation function
    Input and output matrix both have shape (batch_size, n_neurons_current_layer)
    """
    return dendritic_potentials


def linear_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    raise NotImplementedError
