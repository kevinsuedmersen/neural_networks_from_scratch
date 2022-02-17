import logging
import math
from typing import List, Tuple, Generator, Union, Dict

import numpy.typing as npt

import src.constants as c
from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses import Loss
from src.lib.metrics import Metric
from src.lib.models import Model
from src.lib.optimizers import Optimizer
from src.utils import log_progress

logger = logging.getLogger(__name__)


class SequentialModel(Model):
    def __init__(
            self,
            loss: Loss,
            metrics_train: List[Metric],
            metrics_val: List[Metric],
            optimizer: Optimizer
    ):
        """
        Instantiates a model consisting of a sequential stack of layers
        :param layers: List of layers from [0, L], where layer 0 represents the input layer and L the output layer
        :param loss: Loss instance computing losses, cost and initializing backpropagation
        :param metrics_train: List of metrics to be evaluated on the training set
        :param metrics_val: List of metrics to be evaluated on the validation set
        :param optimizer: Optimizer instance applying weight updates
        """
        self.loss = loss
        self.metrics_train = metrics_train
        self.metrics_val = metrics_val
        self.optimizer = optimizer

        self.costs = []
        self.layers: List[Union[InputLayer, DenseLayer]] = []
        self.n_layers = None
        self.history = self._init_history()

    def _init_history(self) -> Dict:
        """Initializes history object"""
        history = {
            c.TRAIN: {metric.name: [] for metric in self.metrics_train},
            c.VAL: {metric.name: [] for metric in self.metrics_val}
        }

        return history

    def add_layer(self, layer: Union[InputLayer, DenseLayer]):
        """Adds a fully initialized layer to the model"""
        # self.layers is empty ==> `layer` is the input layer
        if not self.layers:
            n_neurons_prev = layer.input_shape[1]

        # self.layers contains at least one layer ==> `layer` is hidden or output layer
        else:
            n_neurons_prev = self.layers[-1].output_shape[1]

        layer.init_parameters(n_neurons_prev)
        # Set layer index before appending it, because index = length - 1
        layer.layer_idx = len(self.layers)
        self.layers.append(layer)
        self.n_layers = len(self.layers)
        # Make sure each layer gets access to the optimizer
        layer.optimizer = self.optimizer

    def _forward_propagate_activations(self, x_train: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Propagate activations from layer 0 to layer L"""
        # Init forward prop
        activations = self.layers[0].forward_propagate(x_train)
        dendritic_potentials = None

        # Forward propagate the activations from layer 1 to layer L
        for l in range(1, self.n_layers):
            activations, dendritic_potentials = self.layers[l].forward_propagate(activations)

        return activations, dendritic_potentials

    def _backward_propagate_errors(
            self,
            ytrue_train: npt.NDArray,
            ypred_train: npt.NDArray,
            dendritic_potentials_out: npt.NDArray
    ):
        """Propagate the errors backward from layer L to layer 1"""
        # Init backprop: Intitially, at layer L (the output layer), compute errors and gradients
        errors = self.loss.init_error(
            ytrue=ytrue_train,
            dendritic_potentials_out=dendritic_potentials_out,
            activations_out=ypred_train
        )
        self.layers[-1].errors = errors
        self.layers[-1].compute_weight_gradients(activations_prev=self.layers[-2].activations)
        self.layers[-1].compute_bias_gradients()

        # Continue to backprop the errors from layer with index L-2 (layer before output layer) to layer with
        # index 1 (layer after input layer)
        for l in range((self.n_layers - 2), 0, -1):
            # input errors ==> layer l+1
            # output errors ==> layer l
            errors = self.layers[l].backward_propagate(errors, self.layers[l + 1].weights)
            self.layers[l].compute_weight_gradients(self.layers[l - 1].activations)
            self.layers[l].compute_bias_gradients()

    def _update_parameters(self):
        """Updates the parameters of each layer using gradient descent. Note that parameter updates
        must be done separately after backpropagating all errors, because during backpropagation,
        all parameters must be kept constant. If parameters are changed during backpropagation,
        the error at layer l might be calculated wrongly, because it is also a fucntion of the
        weights in layer l+1 (see line 95 above)
        """
        for layer in self.layers[1:]:
            layer.update_parameters()

    def _evaluate_metrics(
            self,
            metrics: List[Metric],
            dataset: str
    ):
        """Evaluates and caches metrics' results"""
        # Evaluate and cache metrics
        metric_log = []
        for metric in metrics:
            metric_value = metric.result()
            self.history[dataset][metric.name].append(metric_value)
            metric_log.append(f"{metric.name} = {metric_value:.2f}")

        # Log metrics
        metric_logs = ", ".join(metric_log)
        log_msg = f"Results on {dataset} dataset: {metric_logs}"
        logger.info(log_msg)

    @staticmethod
    def _update_metrics(
            ytrue: npt.NDArray,
            ypred: npt.NDArray,
            metrics: List[Metric]
    ):
        """Updates the state of each metric"""
        for metric in metrics:
            metric.update_state(ytrue, ypred)

    def train_step(self, x_train: npt.NDArray, ytrue_train: npt.NDArray):
        """Includes the forward pass, cost computation, backward pass and parameter update"""
        activations_out, dendritic_potentials_out = self._forward_propagate_activations(x_train)
        losses = self.loss.compute_losses(ytrue_train, activations_out)
        cost = self.loss.compute_cost(losses)
        self.costs.append(cost)
        self._backward_propagate_errors(
            ytrue_train=ytrue_train,
            ypred_train=activations_out,
            dendritic_potentials_out=dendritic_potentials_out
        )
        self._update_parameters()
        self._update_metrics(ytrue_train, activations_out, self.metrics_train)

    def val_step(self, x_val: npt.NDArray, ytrue_val: npt.NDArray):
        """Updates metrics' states on the validation set"""
        activations_out, _ = self._forward_propagate_activations(x_val)
        self._update_metrics(ytrue_val, activations_out, self.metrics_val)

    def train(
            self,
            data_gen_train: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            data_gen_val: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            n_epochs: int,
            batch_size: int,
            n_samples_train: int = None,
            n_samples_val: int = None,
            logging_frequency: int = 100,
            **kwargs
    ):
        """Trains the multi-layer perceptron batch-wise for ``epochs`` epochs
        """
        n_batches_train = math.ceil(n_samples_train / batch_size)
        n_batches_val = math.ceil(n_samples_val / batch_size)

        logger.info("Training started")
        for epoch_counter in range(n_epochs):
            # Train on batches of training data until there is no data left
            for train_batch_counter, (x_train, ytrue_train) in enumerate(data_gen_train):
                self.train_step(x_train, ytrue_train)
                log_progress(train_batch_counter, n_batches_train, "Training on batches")

            # Evaluate on the validation set
            for val_batch_counter, (x_val, ytrue_val) in enumerate(data_gen_val):
                self.val_step(x_val, ytrue_val)
                log_progress(val_batch_counter, n_batches_val, "Validating on batches")

            # Evaluate metrics once per epoch
            self._evaluate_metrics(self.metrics_train, c.TRAIN)
            self._evaluate_metrics(self.metrics_val, c.VAL)
            log_progress(epoch_counter, n_epochs, "Epoch completed")

    def predict(self, x: npt.NDArray, **kwargs) -> npt.NDArray:
        pass

    def evaluate(
            self,
            data_gen: Generator[Tuple[npt.NDArray, npt.NDArray], None, None],
            **kwargs
    ):
        pass
