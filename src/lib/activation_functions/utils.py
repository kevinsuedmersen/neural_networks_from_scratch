from typing import Tuple

import numpy as np
import numpy.typing as npt


def init_sigmoid_tanh(dendritic_potentials: npt.NDArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to initialize the sigmoid or tanh activation functions"""
    # Get the indices of the positive/negative inputs
    positive = dendritic_potentials >= 0
    negative = dendritic_potentials < 0

    # Init activations
    activations = np.zeros(dendritic_potentials.shape)

    return positive, negative, activations
