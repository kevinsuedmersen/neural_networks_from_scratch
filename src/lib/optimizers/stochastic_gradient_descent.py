import logging

import numpy.typing as npt

from src.lib.optimizers import Optimizer

logger = logging.getLogger(__name__)


class StochasticGradientDescentOptimizer(Optimizer):
    def update_parameters(self, parameters: npt.NDArray, gradients: npt.NDArray):
        """Applies stochastic gradient descent updates on `parameters`"""
        new_parameters = parameters - (self.learning_rate * gradients)

        return new_parameters
