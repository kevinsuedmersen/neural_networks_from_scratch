import logging

import numpy as np

from src.lib.data_generators import DataGenerator
from src.lib.data_generators.image import ImageDataGenerator

logger = logging.getLogger(__name__)


def get_data_generator(
        data_gen_name: str,
        data_dir: str,
        val_size: float,
        test_size: float,
        batch_size: int,
        img_height: int = None,
        img_width: int = None,
        random_state: int = None
) -> DataGenerator:
    """Returns a train, validation and test data_generators generators"""
    if random_state is not None:
        np.random.seed(random_state)

    if data_gen_name == "image":
        data_gen = ImageDataGenerator(
            data_dir,
            val_size,
            test_size,
            batch_size,
            img_height,
            img_width
        )
    else:
        raise NotImplementedError(f"Data generator '{data_gen_name}' has not been implemented yet")

    return data_gen
