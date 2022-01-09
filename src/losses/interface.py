import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

from src.activation_functions import get_jacobian_function

logger = logging.getLogger(__name__)


class Loss(ABC):
    def __init__(self, output_activation: str, task: str):
        """
        Instantiates a loss object
        :param output_activation: Activation function at the output layer
        :param task: For classifications, it must be either "multi_class_classification" or
            "multi_label_classification". For regressions, it must be "regression"
        """
        assert task in ["multi_class_classification", "multi_label_classification", "regression"]
        self.output_activation = output_activation
        self.task = task
        self.jacobian_function = get_jacobian_function(output_activation)

    @abstractmethod
    def compute_losses(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_cost(self, losses: npt.NDArray):
        pass

    @abstractmethod
    def init_error(self, *args, **kwargs):
        pass