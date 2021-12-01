import logging
from typing import Tuple, Union

import numpy.typing as npt

from src.layers.interface import Layer
from src.types import BatchSize, NNeurons, NNeuronsPrev, NNeuronsNext, One, NFeatures

logger = logging.getLogger(__name__)


class DenseLayer(Layer):
    @staticmethod
    def init_activations(
            x_batch: npt.NDArray[Tuple[BatchSize, ...]]
    ) -> npt.NDArray[Tuple[BatchSize, NFeatures, One]]:
        """Makes sure x_batch is converted into an array of shape (batch_size, n_features, 1)"""
        x_batch_reshaped = x_batch.reshape((x_batch.shape[0], -1, 1))

        return x_batch_reshaped

    def forward(
            self,
            activations_prev_l: npt.NDArray[Tuple[BatchSize, Union[NFeatures, NNeuronsPrev], One]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    def backward(
            self,
            error_next: npt.NDArray[Tuple[BatchSize, NNeuronsNext, One]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeurons, 1]]:
        pass

    def update_params(self):
        pass
