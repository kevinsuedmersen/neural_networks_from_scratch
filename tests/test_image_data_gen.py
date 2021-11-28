import pytest

from src.data_gen.image import ImageDataGenerator
from tests.test_config import TestConfig


class TestImageDataGenerator(TestConfig):
    data_dir = "fixtures"
    val_size = 0.3
    test_size = 0.3
    batch_size = 8
    img_height = 128
    img_width = 128
    img_extension = (".jpg",)

    @pytest.fixture
    def img_data_gen(self) -> ImageDataGenerator:
        img_data_gen = ImageDataGenerator(
            self.data_dir,
            self.val_size,
            self.test_size,
            self.batch_size,
            self.img_height,
            self.img_width,
            self.img_extension
        )
        return img_data_gen

    def test_train(self, img_data_gen):
        train_data_gen = img_data_gen.train()
        for img_batch, label_batch in train_data_gen:
            print(img_batch.shape, label_batch.shape)
