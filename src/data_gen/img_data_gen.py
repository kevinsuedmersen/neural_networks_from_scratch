import logging
import os
from typing import List, Tuple, Generator

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from src.types import BatchSize, ImgHeight, ImgWidth

logger = logging.getLogger(__name__)


def get_abs_img_paths(data_dir: str) -> List[str]:
    abs_img_paths = []
    for filename in os.listdir(data_dir):
        if filename.endswith((".png", ".bmp", ".jpeg", ".jpg")):
            abs_img_path = os.path.join(data_dir, filename)
            abs_img_paths.append(abs_img_path)

    return abs_img_paths


def train_val_test_split(
        abs_img_paths: List[str],
        val_size: float,
        test_size: float
) -> Tuple[List[str], List[str], List[str]]:
    """Splits a list of absolute image filepaths into train, validation and test set"""
    img_paths_train, img_paths_test = train_test_split(abs_img_paths, test_size=test_size)
    img_paths_train, img_paths_val = train_test_split(abs_img_paths, test_size=val_size)

    return img_paths_train, img_paths_val, img_paths_test


def batch_generator(
        abs_img_paths: List[str],
        batch_size: int,
        img_height: int,
        img_width: int
) -> Generator[npt.NDArray[BatchSize, ImgHeight, ImgWidth], None, None]:
    """Creates a generator yielding a batch of images"""
    img_arrays = []
    for counter, img_path in enumerate(abs_img_paths):
        img_array = cv2.imread(img_path)
        img_resized = cv2.resize(img_array, img_height, img_width)
        img_arrays.append(img_resized)
        if (counter + 1) % batch_size == 0:
            batch_array = np.concatenate(img_arrays, axis=0)
            img_arrays = []
            yield batch_array


def img_data_generators(
        data_dir: str,
        val_size: float,
        test_size: float,
        batch_size: int,
        img_height: int,
        img_width: int
) -> Tuple[Generator, Generator, Generator]:
    """Returns a tuple of image data_gen generators for training, validation and testing each of them
    yielding a batch of images
    """
    abs_img_paths = get_abs_img_paths(data_dir)
    img_paths_train, img_paths_val, img_paths_test = train_val_test_split(
        abs_img_paths,
        val_size,
        test_size
    )
    data_gen_train = batch_generator(img_paths_train, batch_size, img_height, img_width)
    data_gen_val = batch_generator(img_paths_val, batch_size, img_height, img_width)
    data_gen_test = batch_generator(img_paths_test, batch_size, img_height, img_width)

    return data_gen_train, data_gen_val, data_gen_test
