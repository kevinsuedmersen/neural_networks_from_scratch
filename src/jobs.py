import logging

import src.constants as c
from src.config_parser import ConfigParser
from src.lib.data_generators.factory import get_data_generator
from src.model_architectures import get_model

logger = logging.getLogger(__name__)


class MLJob:
    def __init__(self, cp: ConfigParser):
        self.cp = cp
        self.data_gen_train = None
        self.data_gen_test = None
        self.model = None

    def train_and_evaluate(self):
        """Trains and evaluates the model"""
        # If only a training data dir is provided, use it for training, validation and testing
        self.data_gen_train = get_data_generator(
            data_gen_name=self.cp.data_gen_name,
            data_dir=self.cp.data_dir_train,
            val_size=self.cp.val_size,
            test_size=self.cp.test_size,
            batch_size=self.cp.batch_size,
            img_height=self.cp.img_height,
            img_width=self.cp.img_width,
            random_state=c.RANDOM_STATE
        )
        data_gen_train, n_samples_train = self.data_gen_train.train()
        data_gen_val, n_samples_val = self.data_gen_train.val()
        data_gen_test, n_samples_test = self.data_gen_train.test()
        n_classes = self.data_gen_train.get_n_classes()

        # If also a testing dir is provided, use it for testing
        if self.cp.data_dir_test is not None:
            self.data_gen_test = get_data_generator(
                data_gen_name=self.cp.data_gen_name,
                data_dir=self.cp.data_dir_test,
                val_size=0,
                test_size=1,
                batch_size=self.cp.batch_size,
                img_height=self.cp.img_height,
                img_width=self.cp.img_width,
                random_state=c.RANDOM_STATE
            )
            data_gen_test, n_samples_test = self.data_gen_test.test()

        self.model = get_model(
            model_name=self.cp.model_name,
            img_height=self.cp.img_height,
            img_width=self.cp.img_width,
            n_color_channels=self.cp.n_color_channels,
            random_state=c.RANDOM_STATE,
            n_classes=n_classes
        )
        self.model.train(
            data_gen_train=data_gen_train,
            data_gen_val=data_gen_val,
            n_epochs=self.cp.n_epochs,
            batch_size=self.cp.batch_size,
            n_samples_train=n_samples_train,
            n_samples_val=n_samples_val
        )
        self.model.evaluate(data_gen_train)
        self.model.evaluate(data_gen_test)
