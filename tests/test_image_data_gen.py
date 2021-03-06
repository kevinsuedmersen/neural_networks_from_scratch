import os
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.lib.data_generators.image import ImageDataGenerator
from tests.test_config import TestConfig


class TestImageDataGenerator(TestConfig):
    """Testing the ImageDataGenerator in a multi-class-classification scheme"""
    data_dir = os.path.join("tests", "fixtures", "cats_vs_dogs")
    val_size = 0.3
    test_size = 0.3
    batch_size = 8
    img_height = 128
    img_width = 128
    img_format = "rgb"
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
            self.img_format,
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

    @pytest.fixture
    def img_paths_2_one_hot(self, img_data_gen, img_paths_2_indices):
        img_paths_2_one_hot = img_data_gen._convert_indices_2_one_hot(img_paths_2_indices)

        return img_paths_2_one_hot

    def test_train_val_test_split(self, img_data_gen, img_paths_2_one_hot):
        img_paths_2_one_hot_train, img_paths_2_one_hot_val, img_paths_2_one_hot_test = img_data_gen._train_val_test_split(
            img_paths_2_one_hot
        )
        # Make sure that no sample was lost
        assert (len(img_paths_2_one_hot_train)
                + len(img_paths_2_one_hot_val)
                + len(img_paths_2_one_hot_test)) == len(img_paths_2_one_hot)

        # Make sure that there is no overlap, i.e. combine all subsets and test that each element is unique
        all_filepaths = [x[0] for x in img_paths_2_one_hot_train] + \
                        [x[0] for x in img_paths_2_one_hot_val] + \
                        [x[0] for x in img_paths_2_one_hot_test]
        # Note that the set function doesn't work with numpy arrays, so I extracted the filepaths
        # from all mappings
        unique_filepaths = set(all_filepaths)
        assert len(unique_filepaths) == len(all_filepaths)

    def test_train(self, img_data_gen):
        train_data_gen = img_data_gen.train()
        for img_batch, label_batch in train_data_gen:
            assert img_batch.shape[1:] == (self.img_height, self.img_width, self.n_channels)
            assert label_batch.shape[1:] == (self.n_labels, 1)
