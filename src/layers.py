import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from src.types import BatchSize, NFeatures, NNeurons, NNeuronsPrev, NNeuronsNext

logger = logging.getLogger(__name__)


class InputLayer:
    def init_activations(
            self,
            x_batch: npt.NDArray[Tuple[BatchSize, NFeatures]]
    ) -> npt.NDArray[Tuple[BatchSize, NFeatures, 1]]:
        pass


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


class DenseLayer(Layer):
    def forward_prop(
            self,
            activations_prev: npt.NDArray[Tuple[BatchSize, NNeuronsPrev, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    def activate(
            self,
            dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    def backward_prop(
            self,
            error_next: npt.NDArray[Tuple[BatchSize, NNeuronsNext, 1]],
            activations: npt.NDArray[Tuple[BatchSize, NNeurons, 1]],
            dendritic_potentials: npt.NDArray[Tuple[BatchSize, NNeurons, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass
