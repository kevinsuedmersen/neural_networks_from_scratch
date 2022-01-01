import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

from src.activation_functions import get_jacobian_function

logger = logging.getLogger(__name__)


class Loss(ABC):
    def __init__(self, activation_function_name: str):
        """
        Instantiates a loss object
        :param activation_function_name: Activation function at the output layer
        """
        self.activation_function_name = activation_function_name
        self.jacobian_function = get_jacobian_function(activation_function_name)

    @abstractmethod
    def compute_losses(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_cost(self, losses: npt.NDArray):
        pass

    @abstractmethod
    def init_error(self, *args, **kwargs):
        pass
