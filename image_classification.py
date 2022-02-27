import logging

from src.config_parser import ImageClassificationConfigParser
from src.jobs import MLJob
from src.utils import get_cli_args, set_root_logger

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    set_root_logger()
    args = get_cli_args()
    cp = ImageClassificationConfigParser(args.config_path)
    ml_job = MLJob(cp)
    ml_job.train_and_evaluate()

