from src.activation_functions.linear import linear_forward, linear_backward
from src.activation_functions.relu import relu_forward, relu_backward
from src.activation_functions.sigmoid import sigmoid_forward, sigmoid_backward
from src.activation_functions.softmax import softmax_forward, softmax_backward, softmax_jacobian
from src.activation_functions.tanh import tanh_forward, tanh_backward


def get_activation_function(activation_function_name: str):
    """Maps a string to an activation function"""
    if activation_function_name == "linear":
        return linear_forward, linear_backward

    if activation_function_name == "relu":
        return relu_forward, relu_backward

    elif activation_function_name == "sigmoid":
        return sigmoid_forward, sigmoid_backward

    elif activation_function_name == "tanh":
        return tanh_forward, tanh_backward

    elif activation_function_name == "softmax":
        return softmax_forward, softmax_backward

    else:
        raise ValueError(f"Unknown activation_function_name provided: {activation_function_name}")


def get_jacobian_function(activation_function_name: str):
    if activation_function_name == "softmax":
        return softmax_jacobian
    else:
        raise ValueError(f"Unknown activation_function_name provided: {activation_function_name}")
