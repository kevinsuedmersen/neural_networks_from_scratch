import logging
from typing import List, Union, Tuple

import numpy.typing as npt

from src.data_gen.interface import DataGenerator
from src.layers.dense import DenseLayer
from src.layers.input import InputLayer
from src.losses.interface import Loss
from src.metrics.interface import Metric
from src.models.interface import Model
from src.optimizers.interface import Optimizer
from src.types import BatchSize, NNeuronsOut, NFeatures, NSamples

logger = logging.getLogger(__name__)


class MultiLayerPerceptron(Model):
    def __init__(
            self,
            layers: List[Union[InputLayer, DenseLayer]],
            loss: Loss,
            metrics_train: List[Metric],
            metrics_val: List[Metric],
            optimizer: Optimizer,
    ):
        """
        Instantiates a Multi Layer Perceptron model
        :param layers: List of layers from [0, L], where layer 0 represents the input layer and L the output layer
        :param loss:
        :param metrics_train:
        :param optimizer:
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.loss = loss
        self.metrics_train = metrics_train
        self.metrics_val = metrics_val
        self.optimizer = optimizer

        self.activations = self._init_cache()
        self.errors = self._init_cache()
        self.costs = []

    def _validate_params(self):
        """Validates parameters"""
        # Make sure that the first layer is an InputLayer
        if not isinstance(self.layers[0], InputLayer):
            raise ValueError(f"The first layer must be an InputLayer instance")

    def _init_cache(self) -> List[Union[None, npt.NDArray]]:
        """Init caches so that their indices correspond to layer indices,
        starting at layer 0 and ending at layer L
        """
        return [None for _ in range(self.n_layers)]

    def _forward_pass(
            self,
            x_train: npt.NDArray[Tuple[BatchSize, ...]],
            ytrue_train: npt.NDArray[Tuple[BatchSize, NNeuronsOut]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeuronsOut]]:
        """Propagate activations from layer 0 to layer L"""
        # Init forward prop
        self.activations[0] = self.layers[0].init_activations(x_train)

        # Forward propagate the activations from layer 1 to layer L
        for l in range(1, self.n_layers):
            self.activations[l] = self.layers[l].forward(self.activations[l - 1])

        return self.activations[-1]

    def _compute_cost(
            self,
            ytrue_batch: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]],
            ypred_batch: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]],
            batch_idx: int
    ):
        losses = self.loss.compute_loss(ytrue_batch, ypred_batch)
        cost = self.loss.compute_cost(losses)
        self.costs.append(cost)
        logger.info(f"Cost after {batch_idx + 1} batches: {cost:.3f}")

    def _backward_pass(self, ytrue_batch: npt.NDArray):
        """Propagate the error backward from layer L to layer 1
        """
        # Init backprop: Compute error at layer L, the output layer
        self.errors[-1] = self.loss.init_error(ytrue_batch, self.activations[1])

        # Backprop the error from layer L-1 to layer 1
        for l in range(self.n_layers - 1, 0, -1):
            self.errors[l] = self.layers[l].backward(self.errors[l + 1])

    def _update_params(self):
        pass

    def _train_step(self, *args, **kwargs) -> npt.NDArray:
        """Includes the forward pass, cost computation, backward pass and parameter update"""
        pass

    def _val_step(
            self,
            x_val: npt.NDArray[BatchSize, NFeatures],
            ytrue_val: npt.NDArray[BatchSize, NFeatures]
    ) -> npt.NDArray[Tuple[BatchSize, NNeuronsOut]]:
        pass

    def _update_metric_state(
            self,
            ytrue: npt.NDArray[Tuple[BatchSize, NNeuronsOut]],
            ypred: npt.NDArray[Tuple[BatchSize, NNeuronsOut]],
            dataset: str
    ):
        pass

    def _get_metric_result(self, dataset: str):
        pass

    def train(self, data_gen: DataGenerator, epochs: int):
        """Trains the multi-layer perceptron batch-wise for ``epochs`` epochs
        """
        for epoch_counter in range(epochs):
            # Train on batches of training data until there is no data left
            for x_train, ytrue_train in data_gen.train():
                ypred_train = self._train_step(x_train, ytrue_train)
                self._update_metric_state(ytrue_train, ypred_train, "train")

            # Evaluate on the validation set
            for x_val, ytrue_val in data_gen.val():
                ypred_val = self._val_step(x_val, ytrue_val)
                self._update_metric_state(ytrue_val, ypred_val, "validation")

            # Evaluate and log performance on the train and test sets
            self._get_metric_result("train")
            self._get_metric_result("validation")

    def predict(
            self,
            x: npt.NDArray[Tuple[NSamples, NFeatures]]
    ) -> npt.NDArray[Tuple[NSamples, NNeuronsOut]]:
        pass

    def evaluate(
            self,
            ytrue: npt.NDArray[NSamples, NNeuronsOut],
            ypred: npt.NDArray[NSamples, NNeuronsOut]
    ):
        pass
