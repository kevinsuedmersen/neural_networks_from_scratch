import logging

import numpy as np
import numpy.typing as npt

from src.losses.interface import Loss

logger = logging.getLogger(__name__)


def categorical_cross_entropy_gradient(ytrue: npt.NDArray, activations: npt.NDArray):
    """Computes the gradient of the loss function w.r.t. the activations in layer L, i.e. the
    predictions
    """
    gradient = -(ytrue / activations)

    return gradient


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

    def init_error(
            self,
            ytrue: npt.NDArray,
            dendritic_potentials: npt.NDArray,
            activations: npt.NDArray
    ) -> npt.NDArray:
        """Initializes the error at the output layer
        :param ytrue:
        """
        # TODO: Wrap if else block into decorator
        if self.activation_function_name == "softmax":
            error = ytrue - activations
        else:
            jacobian = self.jacobian_function(dendritic_potentials, activations)
            gradient = categorical_cross_entropy_gradient(ytrue, activations)
            error = np.matmul(jacobian, gradient)

        return error