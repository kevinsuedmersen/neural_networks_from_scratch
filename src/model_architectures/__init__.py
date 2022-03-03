from typing import Union

import numpy as np
import tensorflow as tf

from src.lib.models import Model
from src.model_architectures.custom_models import get_mlp_model
from src.model_architectures.tf_models import get_benchmark_mlp_model


def get_model(
        model_name: str,
        img_height: int = None,
        img_width: int = None,
        n_color_channels: int = None,
        random_state: int = None,
        n_classes: int = None,
        learning_rate: int = None
) -> Union[Model, tf.keras.Sequential]:
    if random_state is not None:
        np.random.seed(random_state)

    if model_name == "mlp":
        return get_mlp_model(img_height, img_width, n_color_channels, n_classes, learning_rate)
    elif model_name == "benchmark_mlp":
        return get_benchmark_mlp_model(img_height, img_width, n_color_channels, n_classes, learning_rate)
    else:
        ValueError(
            f"Unknown or non-implemented model_name provided: {model_name}. "
            f"Available model_names are: ['mlp', benchmark_mlp']"
        )
