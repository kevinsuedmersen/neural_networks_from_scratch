import logging

from src.data_gen.factory import get_data_generator
from src.models.factory import get_model
from src.utils import get_cli_args

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = get_cli_args()
    model = get_model(args.model_name)
    data_gen = get_data_generator(
        args.data_gen_name,
        data_dir=args.data_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    model.train(data_gen, 50)
    model.evaluate(data_gen)
