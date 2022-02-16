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
    """Parses CLI arguments"""
    parser = argparse.ArgumentParser(description="CLI argument parser")
    parser.add_argument("--config_path", default=None, type=str)
    args = parser.parse_args()

    return args


def log_progress(
        counter: int,
        total: int,
        log_msg: str = None
):
    """Logs progress of a long computation"""
    progress = f"(Progress: {counter + 1}/{total} = {(counter + 1)/total*100:.2f}%)"

    if log_msg is not None:
        log_msg = f"{log_msg}. {progress}"
    else:
        log_msg = progress

    logger.info(log_msg)
