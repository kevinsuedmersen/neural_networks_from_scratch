import argparse
import logging

logger = logging.getLogger(__name__)


def set_root_logger():
    """Sets up the root logger
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    logger.info('Root logger is set up')


def get_cli_args():
    parser = argparse.ArgumentParser(description="CLI argument parser")
    parser.add_argument("--config_path", default=None, type=str)
    args = parser.parse_args()

    return args
