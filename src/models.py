import logging
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from src.layers import DenseLayer
from src.losses import Loss
from src.metrics import Metric
from src.optimizers import Optimizer
from src.types import BatchSize, NNeuronsCurr, NNeuronsPrev, NNeuronsOut

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
        self.n_layers = len(layers)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.weighted_inputs: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = self._init_cache()
        self.activations: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = self._init_cache()
        self.errors: List[np.ndarray((BatchSize, Union[NNeuronsPrev, NNeuronsCurr], 1))] = self._init_cache()
        self.costs: List[float] = []

    def _init_cache(self) -> List[None]:
        """Init caches so that their indices correspond to layer indices, starting at layer 0 and ending at layer L"""
        return [None for _ in range(self.n_layers)]

    def _forward_prop(
            self,
            x_batch: np.ndarray((BatchSize, ...)),
            y_batch: np.ndarray((BatchSize, NNeuronsOut))
    ):
        """Propagate activations from layer 0 to layer L and return predictions"""
        # Init forward prop: Preprocess the raw input data
        self.activations[0] = self.layers[0].init_activations(x_batch)

        # Forward propagate the activations from layer 1 to layer L
        for l in range(1, self.n_layers):
            self.weighted_inputs[l] = self.layers[l].forward_prop(self.activations[l - 1])
            self.activations[l] = self.layers[l].activate(self.weighted_inputs[l])

    def _compute_cost(
            self,
            ytrue_batch: np.ndarray((BatchSize, NNeuronsOut)),
            ypred_batch: np.ndarray((BatchSize, NNeuronsOut)),
            batch_idx: int
    ):
        losses = self.loss.compute_loss(ytrue_batch, ypred_batch)
        cost = self.loss.compute_cost(losses)
        self.costs.append(cost)
        logger.info(f"Cost after {batch_idx + 1} batches: {cost:.3f}")

    def _backward_prop(
            self,
            ytrue_batch: np.ndarray((BatchSize, NNeuronsOut)),
            ypred_batch: np.ndarray((BatchSize, NNeuronsOut))
    ):
        """Propagate the error backward from layer L to layer 1"""
        # Init backprop: Compute error at layer L, the output layer
        self.errors[-1] = self.loss.init_error(self.activations[-1], self.weighted_inputs[-1])

        # Backprop the error from layer L-1 to layer 1
        for l in range(self.n_layers - 1, 0, -1):
            self.errors[l] = self.layers[l].backward_prop(self.errors[l+1], self.activations[l], self.weighted_inputs[l])

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
