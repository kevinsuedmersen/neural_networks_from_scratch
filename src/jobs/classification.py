import copy
import logging
import math
from typing import Tuple, Union, List

import tensorflow as tf

import src.constants as c
from src.config_parser.classification import ImageClassificationConfigParser
from src.jobs import MLJob
from src.lib.data_generators import DataGenerator
from src.lib.data_generators.factory import get_data_generator
from src.lib.models import Model
from src.model_architectures import get_model

logger = logging.getLogger(__name__)


class ImageClassificationJob(MLJob):
    def __init__(self, cp: ImageClassificationConfigParser):
        super().__init__(cp)
        self.data_gen_train, self.data_gen_test = self._get_data_generators()
        self.model = self._get_model(self.data_gen_train.n_classes, self.cp.model_name)

    def _get_data_generators(self) -> Tuple[DataGenerator, DataGenerator]:
        data_gen_train = get_data_generator(
            data_gen_name=self.cp.data_gen_name,
            data_dir=self.cp.data_dir_train,
            val_size=self.cp.val_size,
            test_size=self.cp.test_size,
            batch_size=self.cp.batch_size,
            random_state=c.RANDOM_STATE,
            img_format=self.cp.img_format,
            img_height=self.cp.img_height,
            img_width=self.cp.img_width,
        )
        # If no separate test data directory is provided, use part of the training data for testing
        if self.cp.data_dir_test is None:
            data_gen_test = copy.deepcopy(data_gen_train)
        else:
            data_gen_test = get_data_generator(
                data_gen_name=self.cp.data_gen_name,
                data_dir=self.cp.data_dir_test,
                val_size=0,
                test_size=1,
                batch_size=self.cp.batch_size,
                random_state=c.RANDOM_STATE,
                img_format=self.cp.img_format,
                img_height=self.cp.img_height,
                img_width=self.cp.img_width,
            )

        return data_gen_train, data_gen_test

    def _get_model(self, n_classes: int, model_name: str) -> Union[Model, tf.keras.Sequential]:
        model = get_model(
            model_name=model_name,
            random_state=c.RANDOM_STATE,
            n_classes=n_classes,
            learning_rate=self.cp.learning_rate,
            img_height=self.cp.img_height,
            img_width=self.cp.img_width,
            n_color_channels=self.cp.n_color_channels,
        )

        return model

    def train(self):
        """Trains the model"""
        self.model.fit(
            data_gen_train=self.data_gen_train,
            n_epochs=self.cp.n_epochs,
            batch_size=self.cp.batch_size
        )

    def evaluate(self):
        """Evaluates the model on train and test set"""
        self.model.evaluate(self.data_gen_train.train(), "train")
        self.model.evaluate(self.data_gen_test.test(), "test")

    def run_in_inference_mode(self):
        """Lets the model run in inference mode"""
        pass

    @staticmethod
    def _log_benchmark_evaluation_results(metric_names: List[str], metric_values: List[float]):
        """Logs evaluation metric results to console"""
        logs = []
        for metric_name, metric_value in zip(metric_names, metric_values):
            logs.append(f"{metric_name}={metric_value}")

        logs_str = ", ".join(logs)
        logger.info(logs_str)

    def benchmark_performance(self):
        """Benchmark performance with tensorflow"""
        tf_model = self._get_model(self.data_gen_train.n_classes, self.cp.benchmark_model_name)
        train_steps = math.ceil(self.data_gen_train.n_samples_train / self.cp.batch_size)
        val_steps = math.ceil(self.data_gen_train.n_samples_val / self.cp.batch_size)
        self.data_gen_train.loop_forever = True
        self.data_gen_test.loop_forever = True
        tf_model.fit(
            x=self.data_gen_train.train(),
            validation_data=self.data_gen_train.val(),
            epochs=self.cp.n_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )

        metric_values_train = tf_model.evaluate(self.data_gen_train.train(), steps=train_steps)
        self._log_benchmark_evaluation_results(tf_model.metrics_names, metric_values_train)
        metric_values_test = tf_model.evaluate(self.data_gen_train.test(), steps=val_steps)
        self._log_benchmark_evaluation_results(tf_model.metrics_names, metric_values_test)
