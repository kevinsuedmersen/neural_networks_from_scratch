import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt

from src.types import BatchSize, NNeurons, NNeuronsPrev, NNeuronsNext

logger = logging.getLogger(__name__)


class Layer(ABC):
    @abstractmethod
    def forward_prop(
            self,
            activations_prev: npt.NDArray[Tuple[BatchSize, NNeuronsPrev, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    @abstractmethod
    def activate(
            self,
            dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    @abstractmethod
    def backward_prop(
            self,
            error_next: npt.NDArray[Tuple[BatchSize, NNeuronsNext, 1]],
            activations: npt.NDArray[Tuple[BatchSize, NNeurons, 1]],
            dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass
