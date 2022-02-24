import logging

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

        logger.info("Config file parsed")
