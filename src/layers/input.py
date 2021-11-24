import logging
from typing import Tuple

import numpy.typing as npt

from src.types import BatchSize, NFeatures

logger = logging.getLogger(__name__)


class InputLayer:
    @staticmethod
    def init_activations(
            x_batch: npt.NDArray[Tuple[BatchSize, ...]]
    ) -> npt.NDArray[Tuple[BatchSize, NFeatures, 1]]:
        """Makes sure x_batch is converted into an array of shape (batch_size, n_features, 1)"""
        x_batch_reshaped = x_batch.reshape((x_batch.shape[0], -1, 1))

        return x_batch_reshaped
