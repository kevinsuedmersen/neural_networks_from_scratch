import logging

import numpy as np
import numpy.typing as npt

from src.losses.interface import Loss
from src.losses.utils import simplify_init_error

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyLoss(Loss):
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

    @staticmethod
    def _compute_loss_gradient(ytrue: npt.NDArray, activations_out: npt.NDArray):
        """Computes the gradient of the loss function w.r.t. the activations in layer L, i.e. the
        predictions
        """
        loss_gradient = -(ytrue / activations_out)

        return loss_gradient

    @simplify_init_error(output_activation_="softmax", task_="multi_class_classification")
    def init_error(
            self,
            ytrue: npt.NDArray,
            dendritic_potentials: npt.NDArray,
            activations: npt.NDArray
    ) -> npt.NDArray:
        """Initializes the error at the output layer"""
        # TODO: Think of and implement a test for this method
        jacobians = self.jacobian_function(dendritic_potentials, activations)
        gradients = self._compute_loss_gradient(ytrue, activations)
        error = np.matmul(jacobians, gradients)

        return error
