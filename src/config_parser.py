import logging
import os.path

from configobj import ConfigObj

logger = logging.getLogger(__name__)


class ConfigParser:
    def __init__(self, config_path: str):
        """Parses the config file"""
        config = ConfigObj(config_path, file_error=True, unrepr=True)

        self.data_gen_name = config["data"]["generator_name"]
        self.data_dir_train = config["data"]["data_dir_train"]
        self.data_dir_test = config["data"]["data_dir_test"]
        self.val_size = config["training"]["val_size"]
        self.test_size = config["training"]["test_size"]

        self.batch_size = config["training"]["batch_size"]
        self.img_height = config["data"]["image_height"]
        self.img_width = config["data"]["image_width"]
        self.n_color_channels = config["data"]["n_color_channels"]

        self.model_name = config["training"]["model_name"]
        self.n_epochs = config["training"]["n_epochs"]
        self.learning_rate = config["training"]["learning_rate"]

        self._validate_params()
        logger.info("Config file parsed and validated")

    def _validate_params(self):
        """Validates parameters semantically"""
        if self.data_dir_test is not None:
            assert os.path.exists(self.data_dir_test), \
                f"Make sure the test data folder '{self.data_dir_test}' exists"
            assert self.test_size == 0, "If separate training data is provided, set test_size = 0"

        assert os.path.exists(self.data_dir_train), \
            f"Make sure the training data folder '{self.data_dir_train}' exists"
