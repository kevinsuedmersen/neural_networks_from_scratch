import logging
from typing import Tuple

import numpy.typing as npt

from src.types import BatchSize, NFeatures

logger = logging.getLogger(__name__)


class InputLayer:
    def init_activations(
            self,
            x_batch: npt.NDArray[Tuple[BatchSize, NFeatures]]
    ) -> npt.NDArray[Tuple[BatchSize, NFeatures, 1]]:
        pass
