import logging

import numpy as np
import numpy.typing as npt

from src.lib.losses import Loss
from src.lib.losses.decorators import simplify_init_error

logger = logging.getLogger(__name__)


class CategoricalCrossentropyLoss(Loss):
    def __init__(self, output_activation, task, epsilon: float = 1e-15):
        super().__init__(output_activation, task, epsilon)
        assert task in ["multi_class_classification", "multi_label_classification"]

    def compute_losses(self, ytrue: npt.NDArray, ypred: npt.NDArray) -> npt.NDArray:
        """Computes the losses for each training example in the current batch"""
        ypred_clipped = self._clip_ypred(ypred)
        ypred_normalized = self._ensure_normalized(ypred_clipped)
        log_losses = -np.sum(ytrue * np.log(ypred_normalized), axis=1, keepdims=True)

        return log_losses

    def compute_cost(self, losses: npt.NDArray) -> float:
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
            dendritic_potentials_out: npt.NDArray,
            activations_out: npt.NDArray
    ) -> npt.NDArray:
        """
        Initializes the error at the output layer. The error initialization is different for every
        loss function, so I decided to model it as a method of the Loss class
        :param ytrue: Ground truth
        :param dendritic_potentials_out: Dendritic potentials at the output layer
        :param activations_out: Activations at the output layer, i.e. predictions
        :return: Error at the output layer
        """
        jacobians = self.activation_function_backward(dendritic_potentials_out, activations_out)
        gradients = self._compute_loss_gradient(ytrue, activations_out)
        error = np.matmul(jacobians, gradients)

        return error
