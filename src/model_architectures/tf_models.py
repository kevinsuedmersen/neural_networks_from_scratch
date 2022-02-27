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
    mlp.add(tf.keras.layers.Dense(512, activation="tanh", input_shape=(img_height, img_width, n_color_channels)))
    mlp.add(tf.keras.layers.Dense(256, activation="tanh"))
    mlp.add(tf.keras.layers.Dense(128, activation="tanh"))
    mlp.add(tf.keras.layers.Dense(64, activation="tanh"))
    mlp.add(tf.keras.layers.Dense(32, activation="tanh"))
    mlp.add(tf.keras.layers.Dense(16, activation="tanh"))
    mlp.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    mlp.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return mlp
