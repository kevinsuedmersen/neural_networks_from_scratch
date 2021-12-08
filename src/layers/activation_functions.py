import numpy as np
import numpy.typing as npt


def linear_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Computes the forward pass of the linear activation function"""
    return dendritic_potentials


def relu_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Computes the forward pass of the ReLU activation function"""
    activations = np.maximum(dendritic_potentials, 0)
    # TODO: Test that activations.shape == dendritic_potentials.shape

    return activations


def softmax_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    """Computes the forward pass of the softmax activation function"""
    # Subtract max (or any other constant) for numerical stability. It will cancel out,
    # because the max is also used in the denominator, i.e. exp_sum
    exp = np.exp(dendritic_potentials - np.max(dendritic_potentials))
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    activations = exp / exp_sum

    # TODO: Test that each row in activations sums up to 1

    return activations


def relu_backward():
    pass


def get_activation_function(activation_function_name: str):
    """Maps a string to an activation function"""
    if activation_function_name == "linear":
        return linear_forward

    if activation_function_name == "relu":
        return relu_forward

    elif activation_function_name == "sigmoid":
        raise NotImplementedError

    elif activation_function_name == "tanh":
        raise NotImplementedError

    elif activation_function_name == "softmax":
        return softmax_forward
