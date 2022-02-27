import logging
import os.path

from configobj import ConfigObj

logger = logging.getLogger(__name__)


class ConfigParser:
    def __init__(self, config_path: str):
        self.config = ConfigObj(config_path, file_error=True, unrepr=True)


class ImageClassificationConfigParser(ConfigParser):
    def __init__(self, config_path: str):
        """Parses the config file"""
        super().__init__(config_path)

        self.data_gen_name = self.config["data"]["generator_name"]
        self.data_dir_train = self.config["data"]["data_dir_train"]
        self.data_dir_test = self.config["data"]["data_dir_test"]
        self.val_size = self.config["training"]["val_size"]
        self.test_size = self.config["training"]["test_size"]

        self.batch_size = self.config["training"]["batch_size"]
        self.img_height = self.config["data"]["image_height"]
        self.img_width = self.config["data"]["image_width"]
        self.img_format = self.config["data"]["img_format"]
        self.n_color_channels = self._get_n_color_channels()

        self.model_name = self.config["training"]["model_name"]
        self.n_epochs = self.config["training"]["n_epochs"]
        self.learning_rate = self.config["training"]["learning_rate"]

        self._validate_params()
        logger.info("Config file parsed and validated")

    def _get_n_color_channels(self) -> int:
        format_2_channel = {
            "grayscale": 1,
            "rgb": 3
        }

        return format_2_channel[self.img_format]

    def _validate_params(self):
        """Validates parameters semantically"""
        # Ensure data directories exist
        if self.data_dir_test is not None:
            assert os.path.exists(self.data_dir_test), \
                f"Make sure the test data folder '{self.data_dir_test}' exists"
            assert self.test_size == 0, "If separate training data is provided, set test_size = 0"
        assert os.path.exists(self.data_dir_train), \
            f"Make sure the training data folder '{self.data_dir_train}' exists"
