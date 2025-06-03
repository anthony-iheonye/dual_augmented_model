
"""
Classes for processing images and masks for visual attributes model training
"""

import os
import re
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
from skimage.filters import (gaussian, threshold_otsu)
from skimage.measure import label
from skimage.segmentation import find_boundaries

from app.utils import create_directory
from drying_package.directoryFunctions import create_directory


class ImageAndMaskDatasetBuilder:
    """Builds a dataset made of images and their corresponding masks."""

    def __init__(self,
                 images_directory: str,
                 masks_directory: str,
                 image_mask_channels: Tuple[int, int],
                 final_image_shape: Union[Tuple[int, int], tuple, None] = None,
                 crop_image_and_mask: bool = False,
                 crop_dimension: Tuple[int, int, int, int] = None,
                 normalize_image: bool = False,
                 normalization_divisor: Union[int, float] = 255,
                 split_mask_into_channels: bool = False,
                 batch_size: Optional[int] = None,
                 shuffle_buffer_size: Optional[int] = None,
                 prefetch_data: bool = None,
                 cache_directory: Optional[str] = None,
                 ):
        """
        :param images_directory: str - Directory containing images.
        :param masks_directory: str - Directory containing masks.
        :param image_mask_channels: (int, int) - The number of channels for the image and mask data.
        :param final_image_shape: (int, int) The final image shape.
        :param crop_image_and_mask: bool - Whether the image and mask should be cropped.
        :param crop_dimension: (int, int, int, int) A tuple (offset_height, offset_width, target_height, target_width).
        :param normalize_image: bool - Whether the image should be normalized.
        :param normalization_divisor: int or float - Value used to scale pixel intensity (e.g., 255 or 127.5).
        :param split_mask_into_channels: bool - Whether to split mask classes into channels.
        :param batch_size: int or None - Batch size. If None, no batching is applied.
        :param shuffle_buffer_size: int or None - Shuffle buffer size. If None, no shuffling is applied.
        :param prefetch_data: bool - Whether to prefetch data using tf.data.AUTOTUNE.
        :param cache_directory: str - Path to a directory for caching the dataset to disk. If `None`,
        caching is skipped. To cache the dataset in memory, set cache_directory to ''.
        """

        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.crop_dimension = crop_dimension
        self.crop_image_and_mask = crop_image_and_mask

        # Generate filepaths to the images and masks
        self.original_image_paths, self.original_mask_paths = self._get_sorted_filepaths_to_images_and_masks(
            images_directory, masks_directory)

        # Extract image and mask format
        img_path = self.original_image_paths[0]
        mask_path = self.original_mask_paths[0]
        self.image_format = self._get_image_format(image_path=img_path)
        self.mask_format = self._get_image_format(image_path=mask_path)
        # Choose the method for decoding images
        if self.image_format.lower() in ['.jpg', '.jpeg']:
            self.decode_image = tf.image.decode_jpeg
        elif self.image_format.lower() == '.png':
            self.decode_image = tf.image.decode_png
        elif self.image_format.lower() == '.bmp':
            self.decode_image = tf.image.decode_bmp
        else:
            self.decode_image = tf.image.decode_png

        # Choose the method for decoding masks
        if self.mask_format.lower() in ['.jpg', '.jpeg']:
            self.decode_mask = tf.image.decode_jpeg
        elif self.mask_format.lower() == '.png':
            self.decode_mask = tf.image.decode_png
        elif self.mask_format.lower() == '.bmp':
            self.decode_mask = tf.image.decode_bmp
        else:
            self.decode_mask = tf.image.decode_png

        # set the number of channels for the image and mask
        self.image_mask_channels = image_mask_channels
        self.image_channels = self.image_mask_channels[0]
        self.mask_channels = self.image_mask_channels[1]

        # set the initial shape of the images and masks.
        self.image_shape, self.mask_shape = self._get_image_and_mask_shape(image_path=img_path,
                                                                           mask_path=mask_path)

        # Check if images and masks are to be resized and/or cropped.
        self.resize_images = False

        if crop_dimension is not None and crop_image_and_mask:
            self.offset_height = crop_dimension[0]
            self.offset_width = crop_dimension[1]
            self.target_height = crop_dimension[2]
            self.target_width = crop_dimension[3]
        else:
            self.crop_image_and_mask = False

        if final_image_shape is not None:
            self.final_image_shape = final_image_shape + (self.image_channels,)
            self.final_mask_shape = final_image_shape + (self.mask_channels,)
            self.new_image_height = tuple(self.final_image_shape)[0]
            self.new_image_width = tuple(self.final_image_shape)[1]

            if self.crop_image_and_mask:
                if self.target_height != self.new_image_height or self.target_width != self.new_image_width:
                    self.resize_images = True
            else:
                if self.new_image_height != self.image_shape[0] or self.new_image_width != self.image_shape[1]:
                    self.resize_images = True


        elif final_image_shape is None and self.crop_image_and_mask and crop_dimension is not None:
            self.final_image_shape = (self.target_height, self.target_width, self.image_channels)
            self.final_mask_shape = (self.target_height, self.target_width, self.mask_channels)
            self.new_image_height = self.target_height
            self.new_image_width = self.target_width

        else:
            self.final_image_shape = self.image_shape
            self.final_mask_shape = self.mask_shape
            self.new_image_height = self.image_shape[0]
            self.new_image_width = self.image_shape[1]

        self.image_mask_dataset = None
        self.tune = tf.data.experimental.AUTOTUNE
        self.unique_intensities = None

        # Generate class labels for the segmentation masks
        self.split_mask_into_channels = split_mask_into_channels
        if split_mask_into_channels:
            self.get_mask_class_labels()

        self.normalize_image = normalize_image
        self.normalization_divisor = normalization_divisor

        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_data = prefetch_data

        # Set cache directory
        if isinstance(cache_directory, str):
            if cache_directory != '':
                self.cache_directory = create_directory(dir_name=cache_directory,
                                                        return_dir=True,
                                                        overwrite_if_existing=True)
            else:
                self.cache_directory = ''
        else:
            self.cache_directory = None

    @staticmethod
    def sort_filenames(file_paths):
        return sorted(file_paths, key=lambda var: [
            int(x) if x.isdigit() else x.lower() for x in re.findall(r'\D+|\d+', var)
        ])

    def _get_sorted_filepaths_to_images_and_masks(self, images_dir, masks_dir):
        """
        Generates the two lists containing sorted paths of images and masks, respectively.

        :param images_dir: Directory containing image files.
        :param masks_dir: Directory containing mask files.
        :return: Two lists – paths to image files and corresponding mask files.

        """
        image_file_list = os.listdir(path=images_dir)
        mask_file_list = os.listdir(path=masks_dir)
        image_paths = [os.path.join(images_dir, filename) for filename in image_file_list]
        mask_paths = [os.path.join(masks_dir, filename) for filename in mask_file_list]

        # sort the file paths in ascending other
        image_paths = self.sort_filenames(image_paths)
        mask_paths = self.sort_filenames(mask_paths)

        return image_paths, mask_paths

    @staticmethod
    def _get_image_format(image_path):
        return os.path.splitext(image_path)[-1]

    def _get_image_and_mask_shape(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = self.decode_image(image, channels=self.image_channels)

        mask = tf.io.read_file(mask_path)
        mask = self.decode_mask(mask, channels=self.mask_channels)

        return image.shape, mask.shape

    def _set_original_shape(self, image, mask):
        """
        Sets width and height information to the image and mask tensors.
        """
        image.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)
        return image, mask

    def _set_final_shape(self, image, mask):
        """
        Sets width and height information to the image and mask tensors.
        """
        image.set_shape(self.final_image_shape)
        mask.set_shape(self.final_mask_shape)
        return image, mask

    def _read_and_decode_image_and_mask(self, image_path: str, mask_path: str):
        """
        Reads and decodes and image and its corresponding masks.
        :param image_path: (str) The image's filepath
        :param mask_path: (str) The mask's filepath
        :return: (tensors) Image and corresponding mask
        """
        # Read image and mask
        image = tf.io.read_file(image_path)
        mask = tf.io.read_file(mask_path)

        image = self.decode_image(contents=image, channels=self.image_channels)
        mask = self.decode_image(contents=mask, channels=self.mask_channels)
        image, mask = self._set_original_shape(image, mask)
        return image, mask

    def _read_and_decode_mask(self, mask_path: str):
        """
        Reads and decodes a segmentation mask.

        :param mask_path: (str) Path to the segmentation mask file.
        :return: Decoded TF tensor representing the mask.
        """

        # Read image and mask
        mask = tf.io.read_file(mask_path)

        mask = self.decode_mask(contents=mask, channels=self.mask_channels)
        # mask.set_shape(self.mask_shape)
        return mask

    @staticmethod
    def _cast_image_mask_to_uint8(image, mask):
        image = tf.cast(image, tf.uint8)
        mask = tf.cast(mask, tf.uint8)
        return image, mask

    @staticmethod
    def _cast_image_mask_to_float(image, mask):
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return image, mask

    @staticmethod
    def _denormalize_image_mask_to_0_255(image, mask):
        """convert image and mask to uint8. The values in the image and mask are scaled between 0 and 255."""
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.uint8)
        return image, mask

    @staticmethod
    def _normalize_image_mask_to_0_1(image, mask):
        """convert image and mask to float32. The values in the image and mask are scaled between 0 and 1."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
        return image, mask

    @staticmethod
    def _normalize_image_to_0_1(image, mask):
        """convert image to float32. The values in the image are scaled between 0 and 1."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, mask

    def _crop_image_and_mask(self, image, mask):
        """Crops out a portion of the image and mask."""
        # crop image and mask
        if self.crop_image_and_mask and self.crop_dimension is not None:
            image = tf.image.crop_to_bounding_box(image, self.offset_height, self.offset_width,
                                                  self.target_height, self.target_width)

            # mask = tf.expand_dims(mask, axis=-1) if len(mask.shape) == 2 else mask

            mask = tf.image.crop_to_bounding_box(mask, self.offset_height, self.offset_width,
                                                 self.target_height, self.target_width)
        return image, mask

    def _resize_image_and_mask(self, image, mask):
        """Resize the image and mask to the predefined dimension."""
        if self.resize_images:
            image = tf.expand_dims(image, axis=-1) if image.ndim == 2 else image
            image = tf.image.resize(images=image, size=(self.new_image_height, self.new_image_width),
                                    method='bilinear')
            image = tf.reshape(tensor=image, shape=(self.new_image_height, self.new_image_width, self.image_channels))

            mask = tf.expand_dims(mask, axis=-1) if mask.ndim == 2 else mask
            mask = tf.image.resize(images=mask, size=(self.new_image_height, self.new_image_width),
                                   method='nearest')
            mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width, self.mask_channels))

            mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width))

            # The resize operation returns image & mask in float values (eg. 125.2, 233. 4),
            # before augmentation, these pixel values need to be normalized to the range [0 - 1],
            # because the tensorflow.keras augmentation layer only accept values in the normalize range of [0 - 1]. To ensure we correctly normalize , we will first
            # round up the current float pixel intensities to whole numbers using tf.cast(image, tf.uint8).
            image, mask = self._cast_image_mask_to_uint8(image, mask)

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = tf.reshape(tensor=mask, shape=(self.new_image_height, self.new_image_width))
            else:
                mask = tf.math.reduce_sum(mask, axis=-1)

        return image, mask

    def get_mask_unique_intensities(self, mask_path):
        """Return the unique pixel intensity for an image."""
        mask = self._read_and_decode_mask(mask_path=mask_path)
        return np.unique(mask).tolist()

    def get_mask_class_labels(self):
        """Compute the unique pixel intensities on all the masks in the dataset."""
        # check the unique intensities of the first fifty or 10% of the image (the larger number)
        num = max(50, (len(self.original_mask_paths) // 10) + 1)
        img_paths = self.original_mask_paths[:num]

        unique_intensities = set()

        for path in img_paths:
            unique_intensities.update(self.get_mask_unique_intensities(path))
        self.unique_intensities = unique_intensities

    def _split_mask_into_channels(self, image, mask):
        """Split mask into channels."""

        mask = tf.cast(mask, dtype=tf.uint8)
        stack_list = []

        # For each class intensity, generate a binary channel showing pixels belonging to that class
        for intensity in self.unique_intensities:
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[..., 0], tf.constant(intensity, dtype=tf.uint8))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.uint8))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=-1)  # Axis starts from 0, so axis of 2 represents the third axis
        return image, mask

    def _normalize_image(self, image, mask):
        """Normalize the image using the divisor.
        If divisor == 255, scales to [0, 1]; otherwise to [-1, 1]."""
        if self.normalize_image:
            image = tf.cast(image, dtype=tf.float32)
            if self.normalization_divisor != 255:
                image = (image - self.normalization_divisor) / self.normalization_divisor
            else:
                image /= 255.0

        return image, mask

    def _read_crop_resize_image_and_mask(self, image_path: str, mask_path: str):
        """
        Read, crop and resize image and corresponding mask.

        :param image_path: (str) path to an Image
        :param mask_path: (str) path to the image's segmentation mask
        :return: (image_tensor, mask_tensor) after preprocessing. If enabled, mask will be split into class channels..
        """
        image, mask = self._read_and_decode_image_and_mask(image_path=image_path, mask_path=mask_path)
        image, mask = self._crop_image_and_mask(image=image, mask=mask)
        image, mask = self._resize_image_and_mask(image=image, mask=mask)
        image, mask = self._normalize_image(image=image, mask=mask)

        if self.split_mask_into_channels:
            image, mask = self._split_mask_into_channels(image=image, mask=mask)
        return image, mask

    def _get_dataset(self, image_paths, mask_paths):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self._read_crop_resize_image_and_mask, num_parallel_calls=self.tune)

        if self.cache_directory is not None:
            dataset = dataset.cache(filename=self.cache_directory)

        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.batch_size:
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)

        if self.prefetch_data:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.image_mask_dataset = dataset

    def run(self):
        self._get_dataset(self.original_image_paths, self.original_mask_paths)




class VisualAttributesDatasetCreator:
    def __init__(self,
                 train_visual_attributes_json_path: str,
                 val_visual_attributes_json_path: str = None,
                 test_visual_attributes_json_path: str = None,
                 save_directory: str = None,
                 visual_properties: tuple = ('L', 'a', 'b', 'contrast', 'correlation', 'energy', 'entropy',
                                             'homogeneity', 'uniformity', 'equivalent_diameter', 'eccentricity',
                                             'feret_diameter_max', 'filled_area', 'perimeter', 'roundness')):

        """
        Produces normalized tensorflow dataset of the training, validation and test visual attributes.

        :param train_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the training set.
        :param val_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the validation set.
        :param test_visual_attributes_json_path: (str) File path to the json file containing the visual attributes for
            each of the images of the test set.
        :param save_directory: (str) path to a the directory where to save a csv file that contains the statistics
            (mean, standard deviation, median, etc) for of the training set. This file would be used for normalization.
        :param visual_properties: (tuple) a tuple containing the various visual properties of interest. These are the
            ones that would be extracted from the json and used to form the tensorflow dataset as well as the final
            pandas dataframe.
        """
        self.visual_properties = list(visual_properties)

        # Dataframe containing the upper threshold for each of the bins associated with the individual visual properties
        self.upper_thresholds_dict = {}

        # create a dictionary containing individual list of number of samples per bin for each visual parameter
        self.num_of_samples_per_bin = {}

        # dictionary containing weight for each bin associated with the respective visual attributes
        self.bin_weights_dict = {}

        self.train_visual_props_json_path = train_visual_attributes_json_path

        if val_visual_attributes_json_path is not None:
            self.val_visual_props_json_path = val_visual_attributes_json_path
        else:
            self.val_visual_props_json_path = None

        if test_visual_attributes_json_path is not None:
            self.test_visual_props_json_path = test_visual_attributes_json_path
        else:
            self.test_visual_props_json_path = None

        self.train_stats = None
        if save_directory is not None:
            self.save_visual_props_stats = True
            self.save_dir = create_directory(dir_name=save_directory, return_dir=True,
                                             overwrite_if_existing=False)
        else:
            self.save_visual_props_stats = False

        self.train_unnorm_visual_props_dataframe = None
        self.val_unnorm_visual_props_dataframe = None
        self.test_unnorm_visual_props_dataframe = None

        self.train_visual_props_weights_dataframe = None
        self.train_visual_props_weights_dataset = None

        self.train_normalized_visual_props_df = None
        self.val_normalized_visual_props_df = None
        self.test_normalized_visual_props_df = None

        self.train_visual_props_dataset = None
        self.val_visual_props_dataset = None
        self.test_visual_props_dataset = None

        self.mean = None
        self.std = None

    def _read_json_file(self, json_file_path):
        """Read a json file and fills the blank rows with zeros."""
        visual_props = pd.read_json(json_file_path)
        visual_props = visual_props.copy().loc[::, self.visual_properties]
        # visual_props.drop(columns=['image_id'], inplace=True)

        # return a zero value for images with no infocus pea
        visual_props.fillna(value=0, inplace=True)
        return visual_props

    def _compute_bin_weights(self, unnorm_train_dataframe):
        """
        Computes the weights for each bin associated with the individual visual attributes. Before computing the
        weights, this method computes a dataframe of the upper threshold of the bins of all the visual attributes. in
        addition to that, it also computes a dictionary, `self.num_of_samples_per_bin`, which is the number of samples
        within each bin for the individual visual attributes.

        :param unnorm_train_dataframe: pandas dataframe of the un-normalized visual attributes values
        """
        # compute the upper threshold of the individual bins that make up each visual attribute
        for visual_property in self.visual_properties:
            # numpy arrays containing the number of samples per bin, as well as the upper thresholds for each bin
            # associated with a particular visual
            num_of_samples_per_bin, upper_thresholds = np.histogram(a=unnorm_train_dataframe[visual_property],
                                                                    bins='doane')

            # update the dictionary self.num_of_samples_per_bin with array of upper_thresholds for each bin associated
            # with each of the visual attributes
            # change 0s in num_per_bin to 1, so we don't divide by zero when computing weights
            zero_index = (num_of_samples_per_bin == 0)
            num_of_samples_per_bin[zero_index] = 1

            # weights per bin
            weight = num_of_samples_per_bin.max() / num_of_samples_per_bin
            weight = np.log10(weight) + 1
            weight[zero_index] = 0

            self.bin_weights_dict[visual_property] = weight
            self.upper_thresholds_dict[visual_property] = upper_thresholds
            num_of_samples_per_bin[zero_index] = 0
            self.num_of_samples_per_bin[visual_property] = num_of_samples_per_bin

    def _produce_visual_attribute_weights_dataframe(self):
        """
        Produces a dataframe containing sample weights for each of the training example within each column in
        a visual properties dataframe.

        :return: Pandas dataframe containing sample weight for each visual properties value with the
            'visual_props_dataframe'.
        """
        visual_attribute_weights_df = pd.DataFrame(data=None, columns=self.visual_properties)
        #
        # for column in visual_props_dataframe.columns:
        #     bin_threshold = self.upper_thresholds_dict[column]
        #     weights = self.bin_weights_dict[column]
        #
        #     visual_attribute_weights_df[column] = visual_props_dataframe[column].apply(func=self._assign_weight,
        #                                                                                args=(bin_threshold, weights))
        for visual_property in self.visual_properties:
            weight_indices = np.digitize(x=self.train_unnorm_visual_props_dataframe[visual_property],
                                         bins=self.upper_thresholds_dict[visual_property],
                                         right=True) - 1
            visual_attribute_weights_df.loc[::, visual_property] = self.bin_weights_dict[visual_property][
                weight_indices]

        return visual_attribute_weights_df

    def _compute_normalization_statistics(self, dataframe):
        """computer mean and standard deviation of the training set"""
        stats = dataframe.describe().transpose()
        mean = stats['mean']
        std = stats['std']
        self.train_stats = stats
        return mean, std

    def _save_stats(self):
        """
        Saves the stats (mean, std, etc) of the visual attributes dataset, for
        the training, validation and test sets.
        """
        self.train_stats.to_csv(f'{self.save_dir}/training_stats.csv')

    def _normalize_dataframe(self, dataframe):
        # normalize dataframe
        normalized_data = (dataframe - self.mean) / self.std
        return normalized_data

    def _read_json_and_produce_normalized_training_dataframe(self):
        """read and normalizes the values within a training dataframe."""
        self.train_unnorm_visual_props_dataframe = self._read_json_file(
            json_file_path=self.train_visual_props_json_path)

        self._compute_bin_weights(unnorm_train_dataframe=self.train_unnorm_visual_props_dataframe)

        self.train_visual_props_weights_dataframe = self._produce_visual_attribute_weights_dataframe()

        # compute normalization statistics (mean and std)
        self.mean, self.std = self._compute_normalization_statistics(dataframe=self.train_unnorm_visual_props_dataframe)
        self.train_normalized_visual_props_df = self._normalize_dataframe(
            dataframe=self.train_unnorm_visual_props_dataframe)

    def _read_json_and_produce_normalized_val_or_test_dataframe(self):
        """read and normalizes a the values within a validation or test dataframe."""
        if self.val_visual_props_json_path is not None:
            self.val_unnorm_visual_props_dataframe = self._read_json_file(
                json_file_path=self.val_visual_props_json_path)
            self.val_normalized_visual_props_df = self._normalize_dataframe(self.val_unnorm_visual_props_dataframe)

        if self.test_visual_props_json_path is not None:
            self.test_unnorm_visual_props_dataframe = self._read_json_file(
                json_file_path=self.test_visual_props_json_path)
            self.test_normalized_visual_props_df = self._normalize_dataframe(self.test_unnorm_visual_props_dataframe)

    def _produce_train_val_test_datasets(self):
        """Produces the training, validation and test datasets."""
        if self.train_visual_props_json_path is not None:
            self.train_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.train_normalized_visual_props_df.to_dict('list'))

        if self.val_visual_props_json_path is not None:
            self.val_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.val_normalized_visual_props_df.to_dict('list'))

        if self.test_visual_props_json_path is not None:
            self.test_visual_props_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=self.test_normalized_visual_props_df.to_dict('list'))

    def _produce_train_visual_props_weight_dataset(self):
        """Produces the training, validation and test datasets."""
        self.train_visual_props_weights_dataset = tf.data.Dataset.from_tensor_slices(
            tensors=self.train_visual_props_weights_dataframe.to_dict('list'))

    def process_data(self):
        self._read_json_and_produce_normalized_training_dataframe()
        self._read_json_and_produce_normalized_val_or_test_dataframe()
        self._produce_train_val_test_datasets()
        self._produce_train_visual_props_weight_dataset()
        if self.save_visual_props_stats:
            self._save_stats()