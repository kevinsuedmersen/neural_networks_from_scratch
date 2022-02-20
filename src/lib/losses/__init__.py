import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.lib.activation_functions import get_activation_function

logger = logging.getLogger(__name__)


class Loss(ABC):
    def __init__(self, output_activation: str, task: str, epsilon: float):
        """
        Instantiates a loss object
        :param output_activation: Activation function at the output layer
        :param task: For classifications, it must be either "multi_class_classification" or
            "multi_label_classification". For regressions, it must be "regression"
        """
        assert task in ["multi_class_classification", "multi_label_classification", "regression"]
        self.output_activation = output_activation
        self.task = task
        self.epsilon = epsilon
        _, self.activation_function_backward = get_activation_function(output_activation)

    def _clip_ypred(self, ypred: npt.NDArray) -> npt.NDArray:
        """Clip ypred into the range [epsilon, (1-epsilon)] TODO: Test this method"""
        ypred_clipped = np.maximum(self.epsilon, np.minimum(1-self.epsilon, ypred))

        return ypred_clipped

    @abstractmethod
    def compute_losses(self, ytrue: npt.NDArray, ypred: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def compute_cost(self, losses: npt.NDArray) -> float:
        pass

    @abstractmethod
    def init_error(
            self,
            ytrue: npt.NDArray,
            dendritic_potentials_out: npt.NDArray,
            activations_out: npt.NDArray
    ) -> npt.NDArray:
        pass
