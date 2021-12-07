import logging

import numpy.typing as npt

from src.losses.interface import Loss

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy(Loss):
    def compute_losses(self, ytrue: npt.NDArray, ypred: npt.NDArray) -> npt.NDArray:
        pass

    def compute_cost(self, losses: npt.NDArray):
        pass

    def init_error(self, activations_out: npt.NDArray, dendritic_potentials_out: npt.NDArray) -> npt.NDArray:
        pass
