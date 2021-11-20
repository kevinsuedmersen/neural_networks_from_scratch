import logging
from typing import Tuple

import numpy.typing as npt

from src.layers.interface import Layer
from src.types import BatchSize, NNeurons, NNeuronsPrev, NNeuronsNext

logger = logging.getLogger(__name__)


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
