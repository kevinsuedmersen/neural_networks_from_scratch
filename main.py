import logging

from src.utils import get_cli_args

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = get_cli_args()
    model = get_model(args.model_name)
    train_data_gen, test_data_gen = get_train_test_data_gen()
    model.train(train_data_gen)
    model.evaluate(train_data_gen)
    model.evaluate(test_data_gen)
