import logging
import os
from typing import List, Tuple, Generator

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from src.data_gen.interface import DataGenerator
from src.types import BatchSize, ImgHeight, ImgWidth, ImgChannels

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

    def _get_img_paths_2_labels(self) -> List[Tuple[str, str]]:
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
        """
        abs_img_paths = []
        for dirpath, dirnames, filenames in os.listdir(self.data_dir):
            for filename in filenames:
                if filename.endswith(self.img_extensions):
                    abs_img_path = os.path.join(dirpath, filename)
                    label = os.path.basename(dirpath)
                    abs_img_paths.append((abs_img_path, label))

        return abs_img_paths

    def _train_val_test_split(
            self,
            img_paths_2_labels: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Splits a list of absolute image filepaths into train, validation and test set"""
        img_paths_train, img_paths_test = train_test_split(img_paths_2_labels, test_size=self.test_size)
        img_paths_train, img_paths_val = train_test_split(img_paths_2_labels, test_size=self.val_size)

        return img_paths_train, img_paths_val, img_paths_test

    def _batch_generator(
            self,
            img_paths_2_labels: List[Tuple[str, str]]
    ) -> Generator[Tuple[
                    npt.NDArray[Tuple[BatchSize, ImgHeight, ImgWidth, ImgChannels]],
                    npt.NDArray[BatchSize, 1]], None, None]:
        """Creates a generator yielding a batch of images"""
        img_arrays = []
        labels = []
        for counter, (img_path, label) in enumerate(img_paths_2_labels):
            # Collect images and labels self.batch_size times
            img_array = cv2.imread(img_path)
            img_resized = cv2.resize(img_array, self.img_height, self.img_width)
            img_rescaled = img_resized / 255
            # Add batch dimension for concatenation later
            img_rescaled = img_rescaled[np.newaxis, ...]
            img_arrays.append(img_rescaled)
            labels.append(label)

            # After self.batch_size elements have been collected transform them into a numpy array
            if (counter + 1) % self.batch_size == 0:
                img_batch = np.concatenate(img_arrays, axis=0)
                label_batch = np.concatenate(labels)
                img_arrays = []
                labels = []
                yield img_batch, label_batch

    def _get_data_gen(
            self,
            dataset: str
    ) -> Generator[Tuple[npt.NDArray[Tuple[BatchSize, ImgHeight, ImgWidth]], npt.NDArray[BatchSize]], None, None]:
        """Returns a tuple of image data_gen generators for training, validation and testing each of them
        yielding a batch of images
        """
        img_paths_2_labels = self._get_img_paths_2_labels()
        img_paths_2_labels_train, img_paths_2_labels_val, img_paths_2_labels_test = self._train_val_test_split(
            img_paths_2_labels
        )
        if dataset == "train":
            data_gen = self._batch_generator(img_paths_2_labels_train)
        elif dataset == "val":
            data_gen = self._batch_generator(img_paths_2_labels_val)
        elif dataset == "test":
            data_gen = self._batch_generator(img_paths_2_labels_test)
        else:
            raise ValueError(f"Unknown dataset provided: {dataset}")

        return data_gen

    def train(self):
        return self._get_data_gen("train")

    def val(self):
        return self._get_data_gen("val")

    def test(self):
        return self._get_data_gen("test")
