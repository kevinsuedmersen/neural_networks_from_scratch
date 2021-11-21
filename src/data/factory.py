import logging
from typing import Tuple, Generator

from src.data.img_data_gen import img_data_generators

logger = logging.getLogger(__name__)


def get_data_generators(
        data_gen_name: str,
        **kwargs
) -> Tuple[Generator, Generator, Generator]:
    """Returns a train, validation and test data generators"""
    if data_gen_name == "img_data_gen":
        data_generators = img_data_generators(
            kwargs["data_dir"],
            kwargs["val_size"],
            kwargs["test_size"],
            kwargs["batch_size"]
        )
    else:
        NotImplementedError(f"Data generator '{data_gen_name}' has not been implemented yet")

    return data_generators
