import logging
import os
import random
from typing import List, Tuple, Generator, Dict

import cv2
import numpy as np
import numpy.typing as npt

from src.data_gen.interface import DataGenerator
from src.types import BatchSize, ImgHeight, ImgWidth, ImgChannels, NNeuronsOut

logger = logging.getLogger(__name__)


class ImageDataGenerator(DataGenerator):
    def __init__(
            self,
            data_dir: str,
            val_size: float,
            test_size: float,
            batch_size: int,
            img_height: int,
            img_width: int,
            img_extensions: Tuple[str] = (".png", ".bmp", ".jpeg", ".jpg")
    ):
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_extensions = img_extensions

        self.label_2_idx: Dict[str, int] = {}

    def _get_img_paths_2_labels(self) -> Dict[str, List[str]]:
        """Maps absolute image filepaths to their corresponding labels. It is assumed that all images
        in a separate folder belong to the same category. Each of these folders is assumed to be
        located in self.data_dir, e.g.
        --data_dir:
          --category_1
            --img_1.png
            --img_2.png
          --category_2
            --img_1.png
            --img_2.png
          ...
        If an image belongs to multiple categories (Multi-Label-Classification), then it should
        be located in multiple folders
        """
        img_paths_2_labels = {}
        # Create absolute image filepaths of all relevant images
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith(self.img_extensions):
                    img_path = os.path.join(dirpath, filename)

                    # Map image filepaths to labels
                    if img_path not in img_paths_2_labels:
                        img_paths_2_labels[img_path] = []
                    img_paths_2_labels[img_path].append(img_path)

        return img_paths_2_labels

    def _get_label_idx(self, label: str) -> int:
        """Returns the index corresponding to ``label``"""
        if len(self.label_2_idx.values()) == 0:
            # If no mapping has been created yet, initialize it and return the first index
            self.label_2_idx[label] = 0
            return 0

        elif label in self.label_2_idx:
            # If the mapping already exists, return the label's corresponding index
            return self.label_2_idx[label]

        else:
            # If a new mapping needs to be created, retrieve the largest index value, increment it
            # by 1, store the new mapping and return the new index
            old_max_idx = max(self.label_2_idx.values())
            new_max_idx = old_max_idx + 1
            self.label_2_idx[label] = new_max_idx
            return new_max_idx

    def _convert_labels_2_one_hot(
            self,
            img_paths_2_labels: Dict[str, List[str]]
    ) -> List[Tuple[str, List[int]]]:
        """Converts the label name strings into a binary vector and returns the mapping between image
        filepaths and binary vectors as a list of tuples
        """
        img_paths_2_one_hot = []
        for img_path, labels in img_paths_2_labels.items():
            one_hot = []
            for label in labels:
                idx = self._get_label_idx(label)
                one_hot.append(idx)
            img_paths_2_one_hot.append((img_path, one_hot))

        return img_paths_2_one_hot

    def _train_val_test_split(
            self,
            img_paths_2_one_hot: List[Tuple[str, List[int]]],
            shuffle: bool = True
    ) -> Tuple[
            List[Tuple[str, List[int]]],
            List[Tuple[str, List[int]]],
            List[Tuple[str, List[int]]]
    ]:
        """Splits a list of absolute image filepaths into train, validation and test set"""
        # Determine sample sizes
        n_samples = len(img_paths_2_one_hot)
        n_samples_train = round(n_samples * (1 - self.test_size))
        n_samples_train_subset = round(n_samples_train * (1 - self.val_size))

        # Shuffle if desired
        if shuffle:
            random.shuffle(img_paths_2_one_hot)

        # Subset all samples accordingly
        train_subset = img_paths_2_one_hot[:n_samples_train_subset]
        val_set = img_paths_2_one_hot[n_samples_train_subset:n_samples_train]
        test_set = img_paths_2_one_hot[n_samples_train:]

        # TODO: Write test that each set is mutually exclusive and that all sets together are collectively exhaustive, i.e. contain all samples

        return train_subset, val_set, test_set

    def _batch_generator(
            self,
            img_paths_2_one_hot_labels: List[Tuple[str, List[int]]]
    ) -> Generator[Tuple[
                    npt.NDArray[Tuple[BatchSize, ImgHeight, ImgWidth, ImgChannels]],
                    npt.NDArray[BatchSize, 1]], None, None]:
        """Creates a generator yielding a batch of images"""
        img_arrays = []
        one_hot_labels = []
        for counter, (img_path, one_hot_label) in enumerate(img_paths_2_one_hot_labels):
            # Collect images and one_hot_labels self.batch_size times
            img_array = cv2.imread(img_path)
            img_resized = cv2.resize(img_array, self.img_height, self.img_width)
            img_rescaled = img_resized / 255
            # Add batch dimension for concatenation later
            img_rescaled = img_rescaled[np.newaxis, ...]
            img_arrays.append(img_rescaled)
            one_hot_labels.append(one_hot_label)

            # After self.batch_size elements have been collected transform them into a numpy array
            if (counter + 1) % self.batch_size == 0:
                img_batch = np.concatenate(img_arrays, axis=0)
                label_batch = np.concatenate(one_hot_labels)
                img_arrays = []
                one_hot_labels = []
                yield img_batch, label_batch

    def _get_data_gen(
            self,
            dataset: str
    ) -> Generator[Tuple[npt.NDArray[Tuple[BatchSize, ImgHeight, ImgWidth, ImgChannels]],
                         npt.NDArray[Tuple[BatchSize, NNeuronsOut]]], None, None]:
        """Returns a tuple of image data_gen generators for training, validation and testing each of them
        yielding a batch of images with their corresponding one-hot-encoded output vectors
        """
        img_paths_2_labels = self._get_img_paths_2_labels()
        img_paths_2_one_hot = self._convert_labels_2_one_hot(img_paths_2_labels)
        img_paths_2_one_hot_train, img_paths_2_one_hot_val, img_paths_2_one_hot_test = self._train_val_test_split(
            img_paths_2_one_hot
        )
        if dataset == "train":
            data_gen = self._batch_generator(img_paths_2_one_hot_train)
        elif dataset == "val":
            data_gen = self._batch_generator(img_paths_2_one_hot_val)
        elif dataset == "test":
            data_gen = self._batch_generator(img_paths_2_one_hot_test)
        else:
            raise ValueError(f"Unknown dataset provided: {dataset}")

        return data_gen

    def train(self):
        return self._get_data_gen("train")

    def val(self):
        return self._get_data_gen("val")

    def test(self):
        return self._get_data_gen("test")
