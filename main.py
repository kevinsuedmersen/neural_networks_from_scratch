import logging

from configobj import ConfigObj

from src.jobs import MLJob
from src.utils import get_cli_args, set_root_logger

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    set_root_logger()

    # Get parameters
    args = get_cli_args()
    config = ConfigObj(args.config_path, file_error=True, unrepr=True)

    ml_job = MLJob(config)
    ml_job.train_and_evaluate()

