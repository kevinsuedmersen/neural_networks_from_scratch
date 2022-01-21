import os

import numpy as np
import numpy.typing as npt
import pytest

from src.config_parser import ConfigParser
from src.utils import set_root_logger

set_root_logger()


class TestConfig:
    batch_size = 32
    n_neurons_prev = 32
    n_neurons = 16
    n_neurons_next = 8
    activation_function_names = ["linear", "relu", "sigmoid", "tanh", "softmax"]

    @pytest.fixture
    def dendritic_potentials(self) -> npt.NDArray:
        """Random dendritic potentials of layer l"""
        dendritic_potentials_ = np.random.randn(self.batch_size, self.n_neurons, 1)

        return dendritic_potentials_

    @pytest.fixture
    def config_parser(self) -> ConfigParser:
        cp = ConfigParser(os.path.join("..", "configs", "image_classification.ini"))
        cp.data_dir = "fixtures"

        return cp
