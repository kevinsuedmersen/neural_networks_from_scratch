import numpy as np

from src.lib.models import Model
from src.model_architectures.own_models import get_mlp_model


def get_model(
        model_name: str,
        img_height: int = None,
        img_width: int = None,
        n_color_channels: int = None,
        random_state: int = None,
        n_classes: int = None,
        learning_rate: int = None
) -> Model:
    if random_state is not None:
        np.random.seed(random_state)

    if model_name == "simple_mlp":
        return get_mlp_model(img_height, img_width, n_color_channels, n_classes, learning_rate)
    else:
        ValueError(f"Unknown or non-implemented model_name provided: {model_name}")
