import logging

import numpy as np
import numpy.typing as npt

from src.losses.interface import Loss

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy(Loss):
    """Categorical cross entropy loss and cost used form multi-class classification problems"""
    def compute_losses(self, ytrue_batch: npt.NDArray, ypred_batch: npt.NDArray) -> npt.NDArray:
        """Computes the losses for each training example in the current batch"""
        logs = ytrue_batch * np.log(ypred_batch)
        losses = -np.sum(logs, axis=1, keepdims=True)

        return losses

    def compute_cost(self, losses: npt.NDArray):
        """Computes the cost, i.e. average loss over all training examples in the current batch"""
        cost = np.mean(losses, axis=0, keepdims=True)

        return cost.item()

    def init_error(self, activations_out: npt.NDArray, dendritic_potentials_out: npt.NDArray) -> npt.NDArray:
        pass
