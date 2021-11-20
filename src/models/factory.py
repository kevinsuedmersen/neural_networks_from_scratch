import logging

from src.layers import InputLayer, DenseLayer
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy
from src.models.mlp import Model, MultiLayerPerceptron
from src.optimizers import StochasticGradientDescent

logger = logging.getLogger(__name__)


def _simple_multi_layer_perceptron() -> MultiLayerPerceptron:
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
        metrics=[Accuracy()],
        optimizer=StochasticGradientDescent()
    )
    return mlp


def get_model(model_name: str) -> Model:
    if model_name == "simple_multi_layer_perceptron":
        return _simple_multi_layer_perceptron()
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
