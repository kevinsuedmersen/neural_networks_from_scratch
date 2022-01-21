import logging

import numpy as np

from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from src.lib.metrics.metrics import Accuracy
from src.lib.models import Model
from src.lib.models.sequential import SequentialModel
from src.lib.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer

logger = logging.getLogger(__name__)


def get_simple_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int
) -> SequentialModel:
    """Creates a simple Multi Layer Perceptron network"""
    mlp = SequentialModel(
        loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescentOptimizer()
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels), layer_idx=0))
    mlp.add_layer(DenseLayer(512, "tanh", layer_idx=1))
    mlp.add_layer(DenseLayer(256, "tanh", layer_idx=2))
    mlp.add_layer(DenseLayer(128, "tanh", layer_idx=3))
    mlp.add_layer(DenseLayer(64, "tanh", layer_idx=4))
    mlp.add_layer(DenseLayer(32, "tanh", layer_idx=5))
    mlp.add_layer(DenseLayer(16, "tanh", layer_idx=6))
    mlp.add_layer(DenseLayer(2, "softmax", layer_idx=7))

    return mlp


def get_tiny_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int
) -> SequentialModel:
    """Creates a tiny Multi Layer Perceptron network for gradient computation tests"""
    mlp = SequentialModel(
        loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescentOptimizer()
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels), layer_idx=0))
    mlp.add_layer(DenseLayer(32, "tanh", layer_idx=4))
    mlp.add_layer(DenseLayer(2, "softmax", layer_idx=7))

    return mlp


def get_model(
        model_name: str,
        img_height: int = None,
        img_width: int = None,
        n_color_channels: int = None,
        random_state: int = None
) -> Model:
    if random_state is not None:
        np.random.seed(random_state)

    if model_name == "simple_mlp":
        return get_simple_mlp_model(img_height, img_width, n_color_channels)
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
