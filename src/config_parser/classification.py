import logging
import os.path

from src.config_parser import ConfigParser

logger = logging.getLogger(__name__)


class ImageClassificationConfigParser(ConfigParser):
    def __init__(self, config_path: str):
        """Parses the config file"""
        super().__init__(config_path)
        self._parse_config_params()

    def _parse_config_params(self):
        super()._parse_config_params()

        data = self.config["data"]
        self.data_gen_name = data["generator_name"]
        self.data_dir_train = data["data_dir_train"]
        self.data_dir_test = data["data_dir_test"]

        self.img_height = data["image_height"]
        self.img_width = data["image_width"]
        self.img_format = data["img_format"]
        self.n_color_channels = self._get_n_color_channels()

        benchmarking = self.config["benchmarking"]
        self.benchmark_model_name = benchmarking["model_name"]

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
