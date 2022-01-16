import numpy as np
import numpy.typing as npt
import pytest


class TestConfig:
    batch_size = 32
    n_neurons = 16
    activation_function_names = ["linear", "relu", "sigmoid", "tanh", "softmax"]

    @pytest.fixture
    def dendritic_potentials(self) -> npt.NDArray:
        """Random dendritic potentials of layer l"""
        dendritic_potentials_ = np.random.randn(self.batch_size, self.n_neurons, 1)

        return dendritic_potentials_
