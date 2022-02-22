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
        """Clip ypred into the range [epsilon, (1-epsilon)]"""
        ypred_clipped = np.clip(ypred, self.epsilon, (1 - self.epsilon))

        return ypred_clipped

    @staticmethod
    def _ensure_normalized(ypred: npt.NDArray) -> npt.NDArray:
        """Ensure predictions for each sample represent probabilities and add up to 1"""
        ypred_sum = np.sum(ypred, axis=1, keepdims=True)
        ypred_normalized = ypred / ypred_sum

        return ypred_normalized

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
