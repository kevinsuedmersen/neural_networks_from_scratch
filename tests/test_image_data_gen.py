from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.data_gen.image import ImageDataGenerator
from tests.test_config import TestConfig


class TestImageDataGenerator(TestConfig):
    """Testing the ImageDataGenerator in a multi-class-classification scheme"""
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
        """In a multi-class-classification scheme, each filepath is mapped to exactly one label.
        Then, we can assert that the label's name is included in the filepath, because the label
        comes from the directory the image is located in
        """
        img_paths_2_labels = img_data_gen._get_img_paths_2_labels()
        for img_path, labels in img_paths_2_labels.items():
            if len(labels) == 1:  # ==> multi-class-classification
                assert labels[0] in img_path

    @pytest.fixture
    def img_paths_2_labels(self, img_data_gen) -> Dict:
        img_paths_2_labels = img_data_gen._get_img_paths_2_labels()

        return img_paths_2_labels

    def test_convert_labels_2_indices(self, img_data_gen, img_paths_2_labels):
        """For each mapping, look up the label name and assert that is included in the image's filepath
        """
        img_paths_2_indices: List[Tuple] = img_data_gen._convert_labels_2_indices(img_paths_2_labels)
        index_2_label = {index: label for label, index in img_data_gen.label_2_index.items()}
        for img_path, label_indices in img_paths_2_indices:
            if len(label_indices) == 1:
                label = index_2_label[label_indices[0]]
                assert label in img_path

    @pytest.fixture
    def img_paths_2_indices(self, img_data_gen, img_paths_2_labels) -> List[Tuple]:
        img_paths_2_indices = img_data_gen._convert_labels_2_indices(img_paths_2_labels)

        return img_paths_2_indices

    def test_convert_labels_2_one_hot(self, img_data_gen, img_paths_2_indices):
        img_paths_2_one_hot = img_data_gen._convert_indices_2_one_hot(img_paths_2_indices)
        for (_, one_hot), (_, label_indices) in zip(img_paths_2_one_hot, img_paths_2_indices):
            # In a multi-class-classification scheme, we know that there should be only one 1 and
            # zeros elsewhere in ``one_hot``
            if len(label_indices) == 1:
                assert np.sum(one_hot) == 1

            # Make sure that the 1 is at the label_index-position
            for label_index in label_indices:
                assert one_hot[label_index] == 1


    def test_train(self, img_data_gen):
        train_data_gen = img_data_gen.train()
        for img_batch, label_batch in train_data_gen:
            assert img_batch.shape == (self.batch_size, self.img_height, self.img_width, self.n_channels)
            assert label_batch.shape == (self.batch_size, self.n_labels)
