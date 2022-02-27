import tensorflow as tf


def get_benchmark_mlp_model(
        img_height: int,
        img_width: int,
        n_color_channels: int,
        n_classes: int,
        learning_rate: float
) -> tf.keras.Sequential:
    """Creates a simple Multi Layer Percepron model using the keras API for benchmakrking"""
    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.layers.Dense(512), input_shape=())