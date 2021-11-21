import logging

from src.models.factory import get_model
from src.utils import get_cli_args

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = get_cli_args()
    model = get_model(args.model_name)
    train_data_gen, val_data_gen, test_data_gen = get_data_generators(
        args.data_gen_name,
        args.data_dir,
        args.val_size,
        args.test_size,
        args.batch_size
    )
    model.train(train_data_gen, val_data_gen)
    model.evaluate(test_data_gen)
