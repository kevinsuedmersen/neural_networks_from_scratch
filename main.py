import logging

from src.config_parser.factory import get_config_parser
from src.jobs.factory import get_ml_job
from src.utils import get_cli_args, set_root_logger

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    set_root_logger()
    args = get_cli_args()
    cp = get_config_parser(args.config_parser_type, args.config_path)
    ml_job = get_ml_job(args.ml_job_type, cp)
    ml_job.benchmark_performance()
    ml_job.train()
    ml_job.evaluate()
