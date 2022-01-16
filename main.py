import logging

from configobj import ConfigObj

from src.data_generators.factory import get_data_generator
from src.model_architectures import get_model
from src.utils import get_cli_args

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Get parameters
    args = get_cli_args()
    config = ConfigObj(args.config_path, file_error=True, unrepr=True)

    # Get data generators
    data_gen = get_data_generator(
        data_gen_name=config["data"]["generator_name"],
        data_dir=config["data"]["data_dir"],
        val_size=config["training"]["val_size"],
        test_size=config["training"]["test_size"],
        batch_size=config["training"]["batch_size"],
        img_height=config["data"]["image_height"],
        img_width=config["data"]["image_width"]
    )
    data_gen_train, n_samples_train = data_gen.train()
    data_gen_val, n_samples_val = data_gen.val()
    data_gen_test, n_samples_test = data_gen.test()

    # Train and evaluate model
    model = get_model(config["training"]["model_name"])
    model.train(
        data_gen_train=data_gen_train,
        data_gen_val=data_gen_val,
        n_epochs=config["training"]["n_epochs"],
        batch_size=config["training"]["batch_size"],
        n_samples_train=n_samples_train,
        n_samples_val=n_samples_val
    )
    model.evaluate(data_gen_train)
    model.evaluate(data_gen_test)
