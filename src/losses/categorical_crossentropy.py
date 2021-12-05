import logging
from typing import Tuple

import numpy.typing as npt

from src.losses.interface import Loss
from src.types import BatchSize, NNeuronsOut

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy(Loss):
    def compute_loss(
            self,
            ytrue: npt.NDArray[Tuple[BatchSize, NNeuronsOut]],
            ypred: npt.NDArray[Tuple[BatchSize, NNeuronsOut]]
    ) -> npt.NDArray[Tuple[BatchSize]]:
        pass

    def compute_cost(
            self,
            losses: npt.NDArray[Tuple[BatchSize]]
    ):
        pass

    def init_error(
            self,
            activations_out: npt.NDArray[Tuple[BatchSize, NNeuronsOut]],
            dendritic_potentials_out: npt.NDArray[Tuple[BatchSize, NNeuronsOut]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeuronsOut]]:
        pass
