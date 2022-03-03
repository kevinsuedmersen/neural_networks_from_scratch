from abc import ABC, abstractmethod

from configobj import ConfigObj


class ConfigParser(ABC):
    def __init__(self, config_path: str):
        self.config = ConfigObj(config_path, file_error=True, unrepr=True)

    @abstractmethod
    def _parse_config_params(self):
        """Parses config params and saves them as instance variables"""
        # We assume that each config file will have a [trainng] section with the following params
        training = self.config["training"]
        self.model_name = training["model_name"]
        self.batch_size = training["batch_size"]
        self.val_size = training["val_size"]
        self.test_size = training["test_size"]
        self.n_epochs = training["n_epochs"]
        self.learning_rate = training["learning_rate"]
