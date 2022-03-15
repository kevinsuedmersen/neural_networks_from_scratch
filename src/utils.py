import argparse
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def set_root_logger():
    """Sets up the root logger
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s; %(name)s.py:%(lineno)s; %(levelname)s; %(message)s')
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    logger.info('Root logger is set up')


def get_cli_args():
    """Parses CLI arguments"""
    parser = argparse.ArgumentParser(description="CLI argument parser")
    parser.add_argument("--config_parser_type", default=None, type=str)
    parser.add_argument("--ml_job_type", default=None, type=str)
    parser.add_argument("--config_path", default=None, type=str)
    args = parser.parse_args()

    return args


def track_time(function_name: str) -> Callable:
    """Decorator for tracking execution time of some function or method"""
    def _decorator(function) -> Callable:
        """Actual inner, parametrized decorator which is returned by calling `track_time`"""
        def _wrapper(*args, **kwargs):
            """Acutal method wrapper for tracking and executing `function`"""
            logger.info(f"Started executing {function_name}")
            tic = time.time()
            function(*args, **kwargs)
            toc = time.time()
            minutes = (toc - tic) / 60
            logger.info(f"Execution of {function_name} took {minutes:.2f} minutes")

        return _wrapper

    return _decorator
