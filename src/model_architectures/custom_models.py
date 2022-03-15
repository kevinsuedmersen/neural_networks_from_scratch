"""Models implemented from scratch
"""
import logging

from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses.categorical_crossentropy import CategoricalCrossentropyLoss
from src.lib.metrics.cost.categorical_crossentropy import CategoricalCrossentropyMetric
from src.lib.metrics.score.accuracy import Accuracy
from src.lib.metrics.score.precision import Precision
from src.lib.metrics.score.recall import Recall
from src.lib.models.sequential import SequentialModel
from src.lib.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer

logger = logging.getLogger(__name__)


def get_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int,
        n_classes: int,
        learning_rate: float
) -> SequentialModel:
    """Creates a simple Multi Layer Perceptron network"""
    mlp = SequentialModel(
        loss=CategoricalCrossentropyLoss("softmax", "multi_class_classification"),
        metrics_train=[
            CategoricalCrossentropyMetric("categorical_crossentropy_train"),
            Accuracy("accuracy_train", None),
            Precision("precision_train", None),
            Recall("recall_train", None)
        ],
        metrics_val=[
            CategoricalCrossentropyMetric("categorical_crossentropy_val"),
            Accuracy("accuracy_val", None),
            Precision("precision_val", None),
            Recall("recall_val", None)
        ],
        optimizer=StochasticGradientDescentOptimizer(learning_rate=learning_rate)
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels)))
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
        metrics_train=[Accuracy("acc_train", None)],
        metrics_val=[Accuracy("acc_val", None)],
        optimizer=StochasticGradientDescentOptimizer()
    )
    mlp.add_layer(InputLayer(input_shape=(None, img_height, img_width, n_color_channels)))
    mlp.add_layer(DenseLayer(32, "tanh"))
    mlp.add_layer(DenseLayer(2, "softmax"))

    return mlp

