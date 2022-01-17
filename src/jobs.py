import logging

from configobj import ConfigObj

from src.data_generators.factory import get_data_generator
from src.model_architectures import get_model

logger = logging.getLogger(__name__)


class MLJob:
    def __init__(self, config: ConfigObj):
        self.config = config

        self.data_gen = get_data_generator(
            data_gen_name=config["data"]["generator_name"],
            data_dir=config["data"]["data_dir"],
            val_size=config["training"]["val_size"],
            test_size=config["training"]["test_size"],
            batch_size=config["training"]["batch_size"],
            img_height=config["data"]["image_height"],
            img_width=config["data"]["image_width"]
        )
        self.model = get_model(config["training"]["model_name"])

    def train_and_evaluate(self):
        data_gen_train, n_samples_train = self.data_gen.train()
        data_gen_val, n_samples_val = self.data_gen.val()
        data_gen_test, n_samples_test = self.data_gen.test()

        self.model.train(
            data_gen_train=data_gen_train,
            data_gen_val=data_gen_val,
            n_epochs=self.config["training"]["n_epochs"],
            batch_size=self.config["training"]["batch_size"],
            n_samples_train=n_samples_train,
            n_samples_val=n_samples_val
        )
        self.model.evaluate(data_gen_train)
        self.model.evaluate(data_gen_test)
