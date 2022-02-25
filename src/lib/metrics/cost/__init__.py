from typing import Union

import numpy as np
import numpy.typing as npt

from src.lib.losses import Loss
from src.lib.metrics import Metric


class CostMetric(Metric):
    """Metrics reflecting a cost, like mean square error"""
    def __init__(self, name: str):
        super().__init__(name)
        # The loss instance will be filled in one of the child classes
        self.loss: Union[Loss, None] = None
        self.all_losses = None

    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        """Calculates and caches losses"""
        losses = self.loss.compute_losses(ytrue, ypred)
        if self.all_losses is None:
            self.all_losses = losses
        else:
            self.all_losses = np.concatenate([self.all_losses, losses], axis=0)

    def result(self) -> float:
        """Computes and returns the current cost value"""
        cost = self.loss.compute_cost(self.all_losses)

        return cost

    def reset_state(self):
        """Resets internal state"""
        self.all_losses = None
