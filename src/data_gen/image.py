import logging
import os
from typing import List, Tuple, Generator

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from src.data_gen.interface import DataGenerator
from src.types import BatchSize, ImgHeight, ImgWidth

logger = logging.getLogger(__name__)


class ImageDataGenerator(DataGenerator):
    def __init__(
            self,
            data_dir: str,
            val_size: float,
            test_size: float,
            batch_size: int,
            img_height: int,
            img_width: int
    ):
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

    def _get_abs_img_paths(self) -> List[str]:
        abs_img_paths = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith((".png", ".bmp", ".jpeg", ".jpg")):
                abs_img_path = os.path.join(self.data_dir, filename)
                abs_img_paths.append(abs_img_path)

        return abs_img_paths

    def _train_val_test_split(
            self,
            abs_img_paths: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Splits a list of absolute image filepaths into train, validation and test set"""
        img_paths_train, img_paths_test = train_test_split(abs_img_paths, test_size=self.test_size)
        img_paths_train, img_paths_val = train_test_split(abs_img_paths, test_size=self.val_size)

        return img_paths_train, img_paths_val, img_paths_test

    def _batch_generator(
            self,
            abs_img_paths: List[str]
    ) -> Generator[npt.NDArray[BatchSize, ImgHeight, ImgWidth], None, None]:
        """Creates a generator yielding a batch of images"""
        img_arrays = []
        for counter, img_path in enumerate(abs_img_paths):
            img_array = cv2.imread(img_path)
            img_resized = cv2.resize(img_array, self.img_height, self.img_width)
            img_arrays.append(img_resized)
            if (counter + 1) % self.batch_size == 0:
                batch_array = np.concatenate(img_arrays, axis=0)
                img_arrays = []
                yield batch_array

    def _get_data_gen(
            self,
            dataset: str
    ) -> Generator[npt.NDArray[BatchSize, ImgHeight, ImgWidth]]:
        """Returns a tuple of image data_gen generators for training, validation and testing each of them
        yielding a batch of images
        """
        abs_img_paths = self._get_abs_img_paths()
        img_paths_train, img_paths_val, img_paths_test = self._train_val_test_split(
            abs_img_paths
        )
        if dataset == "train":
            data_gen = self._batch_generator(img_paths_train)
        elif dataset == "val":
            data_gen = self._batch_generator(img_paths_val)
        elif dataset == "test":
            data_gen = self._batch_generator(img_paths_test)
        else:
            raise ValueError(f"Unknown dataset provided: {dataset}")

        return data_gen

    def train(self):
        return self._get_data_gen("train")

    def val(self):
        return self._get_data_gen("val")

    def test(self):
        return self._get_data_gen("test")
