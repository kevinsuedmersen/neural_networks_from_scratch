import logging

from configobj import ConfigObj

from src.data_gen.factory import get_data_generator
from src.models.factory import get_model
from src.utils import get_cli_args

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = get_cli_args()
    config = ConfigObj(args.config_path, file_error=True, unrepr=True)
    data_gen = get_data_generator(
        data_gen_name=config["data"]["generator_name"],
        data_dir=config["data"]["data_dir"],
        val_size=config["training"]["val_size"],
        test_size=config["training"]["test_size"],
        batch_size=config["training"]["batch_size"],
        img_height=config["data"]["image_height"],
        img_width=config["data"]["image_width"]
    )
    model = get_model(config["training"]["model_name"])
    model.train(
        data_gen.train(),
        data_gen.val(),
        config["training"]["epochs"],
        config["training"]["batch_size"]
    )
    model.evaluate(data_gen.train())
    model.evaluate(data_gen.test())
