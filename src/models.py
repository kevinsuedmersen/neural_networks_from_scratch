import logging
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from src.layers import DenseLayer
from src.losses import Loss
from src.metrics import Metric
from src.optimizers import Optimizer
from src.types import BatchSize, NNeuronsCurr, NNeuronsPrev

logger = logging.getLogger(__name__)


class Model(ABC):
    @abstractmethod
    def _forward_pass(self):
        pass

    @abstractmethod
    def _compute_cost(self):
        pass

    @abstractmethod
    def _backward_pass(self):
        pass

    @abstractmethod
    def _update_params(self):
        pass

    @abstractmethod
    def _gen_batch(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class MultiLayerPerceptron(Model):
    def __init__(
            self,
            layers: List[DenseLayer],
            loss: Loss,
            metrics: List[Metric],
            optimizer: Optimizer,
    ):
        self.layers = layers
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        # Placeholders
        self.z_cache: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = []
        self.a_cache: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = []
        self.errors: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = []

    def _forward_pass(self):
        pass

    def _compute_cost(self):
        pass

    def _backward_pass(self):
        pass

    def _update_params(self):
        pass

    def _gen_batch(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
