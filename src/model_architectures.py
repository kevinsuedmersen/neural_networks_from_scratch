import logging

from src.layers.dense import DenseLayer
from src.layers.input import InputLayer
from src.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from src.metrics.metrics import Accuracy
from src.models.sequential import Model, SequentialModel
from src.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer

logger = logging.getLogger(__name__)


def get_simple_mlp_model() -> SequentialModel:
    mlp = SequentialModel(
        loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescentOptimizer()
    )
    mlp.add_layer(InputLayer(input_shape=(None, 128, 128, 3), layer_idx=0))
    mlp.add_layer(DenseLayer(512, "relu", layer_idx=1))
    mlp.add_layer(DenseLayer(256, "relu", layer_idx=2))
    mlp.add_layer(DenseLayer(128, "relu", layer_idx=3))
    mlp.add_layer(DenseLayer(64, "relu", layer_idx=4))
    mlp.add_layer(DenseLayer(32, "relu", layer_idx=5))
    mlp.add_layer(DenseLayer(16, "relu", layer_idx=6))
    mlp.add_layer(DenseLayer(2, "softmax", layer_idx=7))

    return mlp


def get_model(model_name: str) -> Model:
    if model_name == "simple_mlp":
        return get_simple_mlp_model()
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
