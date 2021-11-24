import logging

from src.layers.dense import DenseLayer
from src.layers.input import InputLayer
from src.losses.categorical_crossentropy import CategoricalCrossEntropy
from src.metrics.metrics import Accuracy
from src.models.mlp import Model, MultiLayerPerceptron
from src.optimizers.optimizers import StochasticGradientDescent

logger = logging.getLogger(__name__)


def simple_multi_layer_perceptron() -> MultiLayerPerceptron:
    mlp = MultiLayerPerceptron(
        layers=[
            InputLayer(),
            DenseLayer(512, "relu"),
            DenseLayer(256, "relu"),
            DenseLayer(128, "relu"),
            DenseLayer(64, "relu"),
            DenseLayer(10, "softmax")
        ],
        loss=CategoricalCrossEntropy(),
        metrics_train=[Accuracy("acc_train")],
        metrics_val=[Accuracy("acc_val")],
        optimizer=StochasticGradientDescent()
    )
    return mlp


def get_model(model_name: str) -> Model:
    if model_name == "simple_multi_layer_perceptron":
        return simple_multi_layer_perceptron()
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
