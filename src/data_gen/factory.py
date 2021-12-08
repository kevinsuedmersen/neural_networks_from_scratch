import logging

from src.data_gen.image import ImageDataGenerator
from src.data_gen.interface import DataGenerator

logger = logging.getLogger(__name__)


def get_data_generator(data_gen_name: str, **kwargs) -> DataGenerator:
    """Returns a train, validation and test data_gen generators"""
    if data_gen_name == "image":
        data_gen = ImageDataGenerator(
            kwargs["data_dir"],
            kwargs["val_size"],
            kwargs["test_size"],
            kwargs["batch_size"],
            kwargs["img_height"],
            kwargs["img_width"]
        )
    else:
        raise NotImplementedError(f"Data generator '{data_gen_name}' has not been implemented yet")

    return data_gen
