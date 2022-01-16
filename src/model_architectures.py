import logging

from src.layers.dense import DenseLayer
from src.layers.input import InputLayer
from src.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from src.metrics.metrics import Accuracy
from src.models.sequential import Model, SequentialModel
from src.optimizers.optimizers import StochasticGradientDescent

logger = logging.getLogger(__name__)


def simple_mlp() -> SequentialModel:
    mlp = SequentialModel(
        loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescent()
    )
    mlp.add_layer(InputLayer(input_shape=(None, 128, 128, 3)))
    mlp.add_layer(DenseLayer(512, "relu"))
    mlp.add_layer(DenseLayer(256, "relu"))
    mlp.add_layer(DenseLayer(128, "relu"))
    mlp.add_layer(DenseLayer(64, "relu"))
    mlp.add_layer(DenseLayer(32, "relu"))
    mlp.add_layer(DenseLayer(16, "relu"))
    mlp.add_layer(DenseLayer(2, "softmax"))

    return mlp


def get_model(model_name: str) -> Model:
    if model_name == "simple_mlp":
        return simple_mlp()
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
