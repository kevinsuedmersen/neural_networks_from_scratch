import logging

from src.config_parser import ConfigParser
from src.data_generators.factory import get_data_generator
from src.model_architectures import get_model

logger = logging.getLogger(__name__)


class MLJob:
    def __init__(self, cp: ConfigParser):
        self.cp = cp

        self.data_gen = get_data_generator(
            data_gen_name=self.cp.data_gen_name,
            data_dir=self.cp.data_dir,
            val_size=self.cp.val_size,
            test_size=self.cp.test_size,
            batch_size=self.cp.batch_size,
            img_height=self.cp.img_height,
            img_width=self.cp.img_width
        )
        self.model = get_model(self.cp.model_name)

    def train_and_evaluate(self):
        data_gen_train, n_samples_train = self.data_gen.train()
        data_gen_val, n_samples_val = self.data_gen.val()
        data_gen_test, n_samples_test = self.data_gen.test()

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
