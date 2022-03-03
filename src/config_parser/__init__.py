from abc import ABC, abstractmethod

from configobj import ConfigObj


class ConfigParser(ABC):
    def __init__(self, config_path: str):
        self.config = ConfigObj(config_path, file_error=True, unrepr=True)

    @abstractmethod
    def _parse_config_params(self):
        """Parses config params and saves them as instance variables"""
        # We assume that each config file will have a [trainng] section with the following params
        self.model_name = self.config["training"]["model_name"]
        self.batch_size = self.config["training"]["batch_size"]
        self.val_size = self.config["training"]["val_size"]
        self.test_size = self.config["training"]["test_size"]
        self.n_epochs = self.config["training"]["n_epochs"]
        self.learning_rate = self.config["training"]["learning_rate"]
