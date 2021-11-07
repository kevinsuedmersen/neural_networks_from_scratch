import logging
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from src.layers import DenseLayer
from src.losses import Loss
from src.metrics import Metric
from src.optimizers import Optimizer
from src.types import BatchSize, NNeurons, NNeuronsPrev, NNeuronsOut

logger = logging.getLogger(__name__)


class Model(ABC):
    @abstractmethod
    def _forward_prop(self, **kwargs):
        pass

    @abstractmethod
    def _compute_cost(self, **kwargs):
        pass

    @abstractmethod
    def _backward_prop(self, **kwargs):
        pass

    @abstractmethod
    def _update_params(self, **kwargs):
        pass

    @abstractmethod
    def _gen_batch(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass


class MultiLayerPerceptron(Model):
    def __init__(
            self,
            layers: List[DenseLayer],
            loss: Loss,
            metrics: List[Metric],
            optimizer: Optimizer,
    ):
        """
        Instantiats a Multi Layer Perceptron model
        :param layers: List of layers from [0, L], where layer 0 represents the input layer and L the output layer
        :param loss:
        :param metrics:
        :param optimizer:
        """
        self.layers = layers
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        # Init caches so that their indices correspond to layer indices
        self.z_cache: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeurons], 1))] = []
        self.a_cache: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeurons], 1))] = []
        self.error_cache: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeurons], 1))] = [None]
        self.costs: List[float] = []

    def _forward_prop(
            self,
            x_batch: np.ndarray((BatchSize, ...)),
            y_batch: np.ndarray((BatchSize, NNeuronsOut))
    ) -> np.ndarray((BatchSize, NNeuronsOut)):
        """Propagate activations from layer 0 to layer L and return predictions"""
        # Init forward prop
        a_prev = x_batch.copy()

        # Forward propagate the activations from layer 0 to layer L
        a = None
        for layer in self.layers:
            z = layer.forward_prop(a_prev)
            self.z_cache.append(z)
            a = layer.activate(z)
            self.a_cache.append(a)
            a_prev = a.copy()

        return a

    def _compute_cost(
            self,
            ytrue_batch: np.ndarray((BatchSize, NNeuronsOut)),
            ypred_batch: np.ndarray((BatchSize, NNeuronsOut)),
            batch_idx: int
    ):
        losses = self.loss(ytrue_batch, ypred_batch)
        cost = self.loss.compute_cost(losses)
        self.costs.append(cost)
        logger.info(f"Cost after {batch_idx + 1} batches: {cost:.3f}")

    def _backward_prop(
            self,
            ytrue_batch: np.ndarray((BatchSize, NNeuronsOut)),
            ypred_batch: np.ndarray((BatchSize, NNeuronsOut))
    ):
        """Propagate the error backward from layer L to layer 1"""
        # Init backprop: Compute error at layer L
        error_curr = self.loss.init_error(self.a_cache[-1], self.z_cache[-1])  # Error at layer L
        self.error_cache.append(error_curr)  # Insert after the `None`

        # Backprop the error from layer L-1 to layer 1
        for layer in reversed(self.layers[1:-1]):
            error_prev = self.layer.backward_prop(error_curr, a_prev, z_prev)
            self.error_cache.insert(1, error_prev)  # Insert before the `None`
            error_curr = error_prev.copy()

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
