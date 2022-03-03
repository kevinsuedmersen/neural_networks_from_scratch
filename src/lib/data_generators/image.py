import logging
import os
import random
from typing import List, Tuple, Generator, Dict

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from src.lib.custom_exceptions import CaseNotHandledError
from src.lib.data_generators import DataGenerator

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
            img_format: str,
            img_extensions: Tuple[str] = (".png", ".bmp", ".jpeg", ".jpg"),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_format = img_format
        self.img_extensions = img_extensions

        self._validate_args()
        self.label_2_index: Dict[str, int] = {}
        self.img_paths_2_one_hot_train, self.img_paths_2_one_hot_val, self.img_paths_2_one_hot_test \
            = self._prepare_images()
        self.n_samples_train = len(self.img_paths_2_one_hot_train)
        self.n_samples_val = len(self.img_paths_2_one_hot_val)
        self.n_samples_test = len(self.img_paths_2_one_hot_test)
        self.n_classes = len(self.label_2_index)
        logger.info(
            f"Image data generator is set up. "
            f"n_samples_train={self.n_samples_train}, "
            f"n_samples_val={self.n_samples_val}, "
            f"n_samples_test={self.n_samples_test}, "
            f"n_classes={self.n_classes}"
        )

    def _validate_args(self):
        """Validates arguments"""
        # Make sure we have images to train with
        if not os.path.exists(self.data_dir):
            raise ValueError(f"data directory {self.data_dir} does not exist")
        else:
            dirnames = os.listdir(self.data_dir)
            if not dirnames:
                raise ValueError(f"data directory {self.data_dir} is empty")

        # Ensure a supported image format is used
        if self.img_format not in ["rgb", "grayscale"]:
            raise ValueError("Currently only RGB and grayscale images are supported")

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
        be located in multiple folders. NOTE that this assumes that if an image belongs to multiple
        categories, its filename (os.path.basename(absolute_img_filepath)) remains the same for all
        categories.
        """
        img_paths_2_labels = {}
        # Create absolute image filepaths of all relevant images
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith(self.img_extensions):
                    img_path = os.path.join(dirpath, filename)

                    # Map image filepaths to labels
                    if filename not in img_paths_2_labels:
                        img_paths_2_labels[img_path] = []
                    label = os.path.basename(dirpath)
                    img_paths_2_labels[img_path].append(label)

        return img_paths_2_labels

    def _get_label_index(self, label: str) -> int:
        """Returns the index corresponding to ``label``"""
        # If no mapping has been created yet, initialize it and return the first index
        if len(self.label_2_index.values()) == 0:
            self.label_2_index[label] = 0
            return 0

        # If the mapping already exists, return the label's corresponding index
        elif label in self.label_2_index:
            return self.label_2_index[label]

        # If a new mapping needs to be created, retrieve the largest index value, increment it,
        # store the new mapping and return the new index
        else:
            old_max_idx = max(self.label_2_index.values())
            new_max_idx = old_max_idx + 1
            self.label_2_index[label] = new_max_idx
            return new_max_idx

    def _convert_indices_2_one_hot(
            self,
            img_paths_2_indices: List[Tuple[str, List[int]]]
    ) -> List[Tuple[str, npt.NDArray]]:
        """Converts a list of label indices (i.e. labels encoded as integers) into a binary vector,
        whose i-th element equals 1, iff the corresponding image belongs to class i.
        """
        img_paths_2_one_hot = []
        n_labels = len(self.label_2_index)
        for img_path, label_indices in img_paths_2_indices:
            one_hot = np.zeros(n_labels)
            for label_index in label_indices:
                one_hot[label_index] = 1
            img_paths_2_one_hot.append((img_path, one_hot))

        return img_paths_2_one_hot

    def _convert_labels_2_indices(
            self,
            img_paths_2_labels: Dict[str, List[str]]
    ) -> List[Tuple[str, List[int]]]:
        """Converts the label name strings into a binary vector and returns the mapping between image
        filepaths and binary vectors as a list of tuples
        """
        img_paths_2_indices = []
        for img_path, labels in img_paths_2_labels.items():
            label_indices = []
            for label in labels:
                label_index = self._get_label_index(label)
                label_indices.append(label_index)
            img_paths_2_indices.append((img_path, label_indices))

        return img_paths_2_indices

    def _train_val_test_split(
            self,
            img_paths_2_one_hot: List[Tuple[str, npt.NDArray]],
            shuffle: bool = True
    ) -> Tuple[List[Tuple[str, npt.NDArray]], List[Tuple[str, npt.NDArray]], List[Tuple[str, npt.NDArray]]]:
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

    def _prepare_images(self) -> Tuple[List, List, List]:
        """Prepares the data generator once at init time"""
        img_paths_2_labels = self._get_img_paths_2_labels()
        img_paths_2_indices = self._convert_labels_2_indices(img_paths_2_labels)
        img_paths_2_one_hot = self._convert_indices_2_one_hot(img_paths_2_indices)
        img_paths_2_one_hot_train, img_paths_2_one_hot_val, img_paths_2_one_hot_test = \
            self._train_val_test_split(img_paths_2_one_hot)

        return img_paths_2_one_hot_train, img_paths_2_one_hot_val, img_paths_2_one_hot_test

    def _read_img(self, img_path: str) -> npt.NDArray:
        """Reads an image and returns it with dimensions (height, width, color_channels)"""
        img_array = np.array(Image.open(img_path))
        if self.img_format == "grayscale":
            if img_array.ndim == 2:
                img_array = img_array[..., np.newaxis]
            elif (img_array.ndim == 3) and (img_array.shape[2] != 1):
                raise CaseNotHandledError(
                    "If img_format='grayscale' and if img_array.ndim=3, the the image should only "
                    "have 1 color channel"
                )
            else:
                raise CaseNotHandledError(
                    f"Grayscale image has unexpected number of dimensions: {img_array.ndim}"
                )

        return img_array

    def _load_and_preprocess(self, img_path: str) -> npt.NDArray:
        """Loads an image from disk, resizes and rescales it"""
        img_array = self._read_img(img_path)

        # Resize the image if provided dimensions don't match its actual dimensions. Note that there
        # is NO batch dimension at this point
        if not img_array.shape[:-1] == (self.img_height, self.img_width):
            img_array = cv2.resize(img_array, (self.img_height, self.img_width))

        img_array = img_array/255

        return img_array

    def _batch_generator(
            self,
            img_paths_2_one_hot: List[Tuple[str, npt.NDArray]]
    ) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        """Creates a generator yielding a batch of images"""
        img_arrays = []
        one_hot_arrays = []
        for counter, (img_path, one_hot_array) in enumerate(img_paths_2_one_hot):
            # Collect images and one_hot_arrays self.batch_size times
            img_array = self._load_and_preprocess(img_path)
            img_array = img_array[np.newaxis, ...]  # shape=(1, ImgHeight, ImgWidth, ImgChannels)
            img_arrays.append(img_array)
            one_hot_array = one_hot_array[np.newaxis, :, np.newaxis]  # shape=(1, NNeuronsOut, 1)
            one_hot_arrays.append(one_hot_array)

            # After self.batch_size elements have been collected or when img_paths_2_one_hot is
            # exhausted, transform them into a numpy arrays
            if ((counter + 1) % self.batch_size == 0) or ((counter + 1) == len(img_paths_2_one_hot)):
                img_batch = np.concatenate(img_arrays, axis=0)
                label_batch = np.concatenate(one_hot_arrays, axis=0)
                img_arrays = []
                one_hot_arrays = []
                yield img_batch, label_batch
                # TODO: Test that in each batch, each image is unique

    def _get_data_gen(self, dataset: str) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        """Returns a tuple of image data_generators generators for training, validation and testing each of them
        yielding a batch of images with their corresponding one-hot-encoded output vectors
        """
        if dataset == "train":
            data_gen = self._batch_generator(self.img_paths_2_one_hot_train)

        elif dataset == "val":
            data_gen = self._batch_generator(self.img_paths_2_one_hot_val)

        elif dataset == "test":
            data_gen = self._batch_generator(self.img_paths_2_one_hot_test)

        else:
            raise ValueError(f"Unknown dataset provided: {dataset}")

        return data_gen

    def train(self) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        return self._get_data_gen("train")

    def val(self) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        return self._get_data_gen("val")

    def test(self) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        return self._get_data_gen("test")
