import logging

import numpy as np

from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses.categorical_crossentropy import CategoricalCrossentropyLoss
from src.lib.metrics.cost.categorical_crossentropy import CategoricalCrossentropyMetric
from src.lib.metrics.score.accuracy import Accuracy
from src.lib.models import Model
from src.lib.models.sequential import SequentialModel
from src.lib.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer

logger = logging.getLogger(__name__)


def get_simple_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int,
        n_classes: int
) -> SequentialModel:
    """Creates a simple Multi Layer Perceptron network"""
    mlp = SequentialModel(
        loss=CategoricalCrossentropyLoss("softmax", "multi_class_classification"),
        metrics_train=[
            CategoricalCrossentropyMetric("categorical_crossentropy_train"),
            #Accuracy("accuracy_train"),
            #Precision("precision_train"),
            #Recall("recall_train")
        ],
        metrics_val=[
            CategoricalCrossentropyMetric("categorical_crossentropy_val"),
            #Accuracy("accuracy_val"),
            #Precision("precision_val"),
            #Recall("recall_val")
        ],
        optimizer=StochasticGradientDescentOptimizer(learning_rate=0.001)
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels)))
    mlp.add_layer(DenseLayer(512, "tanh"))
    mlp.add_layer(DenseLayer(256, "tanh"))
    mlp.add_layer(DenseLayer(128, "tanh"))
    mlp.add_layer(DenseLayer(64, "tanh"))
    mlp.add_layer(DenseLayer(32, "tanh"))
    mlp.add_layer(DenseLayer(16, "tanh"))
    mlp.add_layer(DenseLayer(n_classes, "softmax"))

    return mlp


def get_tiny_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int
) -> SequentialModel:
    """Creates a tiny Multi Layer Perceptron network for gradient computation tests"""
    mlp = SequentialModel(
        loss=CategoricalCrossentropyLoss("softmax", "multi_class_classification"),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescentOptimizer()
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels)))
    mlp.add_layer(DenseLayer(32, "tanh"))
    mlp.add_layer(DenseLayer(2, "softmax"))

    return mlp


def get_model(
        model_name: str,
        img_height: int = None,
        img_width: int = None,
        n_color_channels: int = None,
        random_state: int = None,
        n_classes: int = None
) -> Model:
    if random_state is not None:
        np.random.seed(random_state)

    if model_name == "simple_mlp":
        return get_simple_mlp_model(img_height, img_width, n_color_channels, n_classes)
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
