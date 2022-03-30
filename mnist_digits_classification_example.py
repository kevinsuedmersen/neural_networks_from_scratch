from src.lib.data_generators.image import ImageDataGenerator
from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses.categorical_crossentropy import CategoricalCrossentropyLoss
from src.lib.metrics.cost.categorical_crossentropy import CategoricalCrossentropyMetric
from src.lib.metrics.score.accuracy import Accuracy
from src.lib.metrics.score.precision import Precision
from src.lib.metrics.score.recall import Recall
from src.lib.models.sequential import SequentialModel
from src.lib.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer
from src.utils import set_root_logger

# Variables which are used more than once. Can be placed into some config file
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_FORMAT = "grayscale"
BATCH_SIZE = 32

if __name__ == "__main__":
    # Make sure logs are printed to the console
    set_root_logger()

    # Image data generators
    data_gen_train = ImageDataGenerator(
        data_dir="resources/mnist_png/training",
        val_size=0.25,
        test_size=0,  # because test data is in a separate directory
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_format=IMG_FORMAT
    )
    data_gen_test = ImageDataGenerator(
        data_dir="resources/mnist_png/testing",
        val_size=0,  # because test data is in a separate directory
        test_size=1,  # because test data is in a separate directory
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_format=IMG_FORMAT
    )

    # Model architecture
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
        optimizer=StochasticGradientDescentOptimizer(learning_rate=0.1)
    )
    mlp.add_layer(InputLayer(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 1)))
    mlp.add_layer(DenseLayer(32, "tanh"))
    mlp.add_layer(DenseLayer(16, "tanh"))
    mlp.add_layer(DenseLayer(10, "softmax"))

    # Training
    mlp.fit(
        data_gen_train=data_gen_train,
        n_epochs=5,
        batch_size=BATCH_SIZE
    )

    # Evaluation. Note that we need to call data_gen_train and data_gen_test here, because
    # we only want a single evaluation loop. Evaluation results will be logged to the console
    mlp.evaluate(data_gen_train.train(), "train")
    mlp.evaluate(data_gen_test.test(), "test")
