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

    # Test images have 3 color channels and are from 2 different categories
    n_channels = 3
    n_labels = 2

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

    def test_get_img_paths_2_labels(self, img_data_gen):
        """In a Multi-Class-Classification scheme, each filepath is mapped to exactly one label.
        Then, we can assert that the label's name is included in the filepath, because the label
        comes from the directory the image is located in
        """
        img_paths_2_labels = img_data_gen._get_img_paths_2_labels()
        for img_path, labels in img_paths_2_labels.items():
            if len(labels) == 1:  # ==> multi-class-classification
                assert labels[0] in img_path

    def test_train(self, img_data_gen):
        train_data_gen = img_data_gen.train()
        for img_batch, label_batch in train_data_gen:
            assert img_batch.shape == (self.batch_size, self.img_height, self.img_width, self.n_channels)
            assert label_batch.shape == (self.batch_size, self.n_labels)
