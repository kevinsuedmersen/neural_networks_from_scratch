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
        total: int = None,
        frequency: int = 100,
        topic: str = None,
        use_print: bool = False
):
    """Logs progress of a long computation"""
    if (counter + 1) % frequency == 0:
        if topic is not None:
            log_msg = f"{topic}: {counter}/{total}"
        else:
            log_msg = f"Progress: {counter}/{total}"

        if total is not None:
            progress_percentage = (counter + 1) / total * 100
            log_msg += f" ({progress_percentage:.2f}%)"

        if use_print:
            print(log_msg)
        else:
            logger.info(log_msg)
