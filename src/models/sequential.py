import logging
from typing import List, Tuple, Generator, Union

import numpy.typing as npt

from src.layers.dense import DenseLayer
from src.layers.input import InputLayer
from src.losses.interface import Loss
from src.metrics.interface import Metric
from src.models.interface import Model
from src.optimizers.interface import Optimizer
from src.utils import log_progress

logger = logging.getLogger(__name__)


class SequentialModel(Model):
    def __init__(
            self,
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
        self.loss = loss
        self.metrics_train = metrics_train
        self.metrics_val = metrics_val
        self.optimizer = optimizer

        self.costs = []
        self.layers: List[Union[InputLayer, DenseLayer]] = []
        self.n_layers = None

    def add_layer(self, layer: Union[InputLayer, DenseLayer]):
        """Adds a fully initialized layer to the model"""
        # if self.layers is empty
        if not self.layers:
            n_neurons_prev = layer.input_shape[1]

        # if self.layers contains at least one layer
        else:
            n_neurons_prev = self.layers[-1].output_shape[1]

        layer.init_parameters(n_neurons_prev)
        self.layers.append(layer)
        self.n_layers = len(self.layers)

    def _forward_pass(self, x_train: npt.NDArray):
        """Propagate activations from layer 0 to layer L"""
        # Init forward prop
        activations = self.layers[0].forward_propagate(x_train)
        dendritic_potentials = None

        # Forward propagate the activations from layer 1 to layer L
        for l in range(1, self.n_layers):
            activations, dendritic_potentials = self.layers[l].forward_propagate(activations)

        return activations, dendritic_potentials

    def _backward_pass(
            self,
            ytrue_train: npt.NDArray,
            ypred_train: npt.NDArray,
            dendritic_potentials_out: npt.NDArray
    ):
        """Propagate the error backward from layer L to layer 1"""
        # Init backprop: Compute error at layer L, the output layer
        error = self.loss.init_error(
            ytrue=ytrue_train,
            dendritic_potentials_out=dendritic_potentials_out,
            activations_out=ypred_train
        )

        # Backprop the error from layer L-1 to layer 1
        for l in range((self.n_layers - 1), 1, -1):
            # layers[l] ==> layer l-1
            # input error ==> layer l
            # output error ==> layer l-1
            error = self.layers[l - 1].backward_propagate(error, self.layers[l].weights)
            self.layers[l - 1].compute_weight_gradients()
            self.layers[l - 1].compute_bias_gradients()
            self.layers[l - 1].update_parameters()

    def train_step(self, x_train: npt.NDArray, ytrue_train: npt.NDArray):
        """Includes the forward pass, cost computation, backward pass and parameter update"""
        activations_out, dendritic_potentials_out = self._forward_pass(x_train)
        losses = self.loss.compute_losses(ytrue_train, activations_out)
        cost = self.loss.compute_cost(losses)
        self.costs.append(cost)
        self._backward_pass(
            ytrue_train=ytrue_train,
            ypred_train=activations_out,
            dendritic_potentials_out=dendritic_potentials_out
        )

    def val_step(self, x_val: npt.NDArray, ytrue_val: npt.NDArray):
        pass

    def train(
            self,
            data_gen_train: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            data_gen_val: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            n_epochs: int,
            batch_size: int,
            n_samples_train: int = None,
            n_samples_val: int = None,
            **kwargs
    ):
        """Trains the multi-layer perceptron batch-wise for ``epochs`` epochs
        """
        logger.info("Training started")
        for epoch_counter in range(n_epochs):
            # Train on batches of training data until there is no data left
            for train_batch_counter, (x_train, ytrue_train) in enumerate(data_gen_train):
                self.train_step(x_train, ytrue_train)
                log_progress(train_batch_counter, n_samples_train, topic="Train batch")

            # Evaluate on the validation set
            for val_batch_counter, (x_val, ytrue_val) in enumerate(data_gen_val):
                self.val_step(x_val, ytrue_val)
                log_progress(val_batch_counter, n_samples_val, topic="Validation batch")

            log_progress(epoch_counter, n_epochs, 1, "Epoch")

    def predict(self, x: npt.NDArray, **kwargs) -> npt.NDArray:
        pass

    def evaluate(
            self,
            data_gen: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            **kwargs
    ):
        pass
