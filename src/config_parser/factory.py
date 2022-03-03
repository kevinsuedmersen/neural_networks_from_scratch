from typing import Union

from src.config_parser.classification import ImageClassificationConfigParser


def get_config_parser(config_parser_type: str, config_path: str) -> Union[ImageClassificationConfigParser]:
    """Gets an instance of a ConfigParser"""
    parsers = {
        "image_classification_config_parser": ImageClassificationConfigParser(config_path)
    }
    return parsers[config_parser_type]
