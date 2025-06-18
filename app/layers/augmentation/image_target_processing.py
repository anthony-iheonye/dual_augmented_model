from typing import List
from typing import List
from typing import Tuple, Union, Optional, Dict, Type, Any

import keras
import numpy as np
import tensorflow as tf
from keras import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from tensorflow.keras.layers import Layer
from tensorflow.tools.docs import doc_controls

H_AXIS = -3
W_AXIS = -2

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    """Checks that the supplied fill mode and interpolation are correct."""
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            "Unknown `fill_mode` {}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported.".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )

def transform(images,
              transforms,
              fill_mode="reflect",
              fill_value=0.0,
              interpolation="bilinear",
              output_shape=None,
              name=None):
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)` (NHWC). The rank
        must be statically known (the shape is not `TensorShape(None)`).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
        b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
        *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
        to the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      output_shape: Output dimension after the transform, `[height, width]`.
        If `None`, output is the same size as input image.
      name: The name of the op.

    Fill mode behavior for each valid value is as follows:

    - reflect (d c b a | a b c d | d c b a)
    The input is extended by reflecting about the edge of the last pixel.

    - constant (k k k k | a b c d | k k k k)
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.

    - wrap (a b c d | a b c d | a b c d)
    The input is extended by wrapping around to the opposite edge.

    - nearest (a a a a | a b c d | d d d d)
    The input is extended by the nearest pixel.

    Input shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.

    Output shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with backend.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                "{}".format(output_shape)
            )

        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )

def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
        given to operation `image_projective_transform_v2`. If one row of
        transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the
        *output* point `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "rotation_matrix"):
        x_offset = (
                           (image_width - 1)
                           - (
                                   tf.cos(angles) * (image_width - 1)
                                   - tf.sin(angles) * (image_height - 1)
                           )
                   ) / 2.0
        y_offset = (
                           (image_height - 1)
                           - (
                                   tf.sin(angles) * (image_width - 1)
                                   + tf.cos(angles) * (image_height - 1)
                           )
                   ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )

def check_input(inputs: Union[int, float, list, tuple], name: str = None,
                input_range: Union[list, tuple] = (0, 200)):
    """Checks if an input is of a certain type and within a specified range."""

    # confirm that inputs and name are not None
    if all(value is not None for value in (inputs, name)):
        # ensure that input_range is a list or tuple
        if isinstance(input_range, (list, tuple)) and len(input_range) == 2:
            min_value = input_range[0]
            max_value = input_range[1]

        else:
            raise ValueError(
                f"`input_range` must be a list or tuple containing "
                f"two values, received {input_range} of type {type(input_range)}")

        if not isinstance(name, str):
            raise TypeError(f"`name` must be a string, received {name} of type {type(name)}.")

        if isinstance(inputs, (int, float)):
            if inputs > max_value or inputs < min_value:
                raise ValueError(
                    f"{name} must be a value between [{min_value}, {max_value}], "
                    f"received {inputs}.")

        elif isinstance(inputs, (list, tuple)):
            inputs = sorted(inputs)
            if inputs[1] > max_value or inputs[0] < min_value:
                raise ValueError(
                    f"The input `{name}` must be a value between [{min_value}, {max_value}], "
                    f"received {inputs}.")
        else:
            raise TypeError("inputs must be a int, float, list, or tuple, receive {}".format(type(inputs)))
    else:
        raise ValueError(
            "`inputs` must either be a [int, float, list or tuple], and "
            "`name` must be a string.")

def center_transformation_matrix(matrices, image_height, image_width):
    """
    Centers a batch of transformation matrices.

    Args:
        matrices: [batch_size, 3, 3] tensor of transform matrices.
        image_height: float or scalar tensor.
        image_width: float or scalar tensor.

    Returns:
        [batch_size, 8] tensor of transforms for ImageProjectiveTransformV3
    """
    height_offset = image_height / 2.0 - 0.5
    width_offset = image_width / 2.0 - 0.5

    offset = tf.convert_to_tensor([[1.0, 0.0, width_offset],
                                   [0.0, 1.0, height_offset],
                                   [0.0, 0.0, 1.0]], dtype=tf.float32)

    reset = tf.convert_to_tensor([[1.0, 0.0, -width_offset],
                                  [0.0, 1.0, -height_offset],
                                  [0.0, 0.0, 1.0]], dtype=tf.float32)

    offset = tf.expand_dims(offset, 0)  # [1, 3, 3]
    reset = tf.expand_dims(reset, 0)    # [1, 3, 3]

    # Apply: M_centered = offset @ matrix @ reset (batched matmul)
    centered = tf.matmul(offset, tf.matmul(matrices, reset))  # [batch, 3, 3]
    flat = tf.reshape(centered[:, :2, :], [-1, 6])  # [batch_size, 6]

    # Pad to 8 by adding [0, 0]
    zeros = tf.zeros((tf.shape(flat)[0], 2), dtype=tf.float32)
    return tf.concat([flat, zeros], axis=1)  # [batch_size, 8]

def get_shear_matrix(angles, image_height, image_width, name=None):
    """Returns a batch of shear matrices for the given angles (in radians)."""
    with tf.name_scope(name or "shear_matrix"):
        angles = tf.convert_to_tensor(angles, dtype=tf.float32)
        batch_size = tf.shape(angles)[0]

        ones = tf.ones_like(angles)
        zeros = tf.zeros_like(angles)
        shear = -tf.sin(angles)
        cos = tf.cos(angles)

        # [batch_size, 3, 3]
        matrices = tf.stack([
            tf.stack([ones, shear, zeros], axis=1),
            tf.stack([zeros, cos, zeros], axis=1),
            tf.stack([zeros, zeros, ones], axis=1)
        ], axis=1)

        return center_transformation_matrix(matrices, image_height, image_width)  # [batch_size, 8]

def interpolate_to_tensorflow_value(custom_value: Union[int, float, list, tuple],
                                    custom_datapoints: Union[list, tuple, None],
                                    tensorflow_datapoints: Union[list, tuple, None]):
    """
    Produces a corresponding TensorFlow value when given a custom value. This is done by
    interpolating between the custom values and their corresponding TensorFlow values.

    :param custom_value: The custom value to be converted Tensorflow equivalent.
    :param custom_datapoints: tuple or list, The   custom data points (x-coordinates),
        must be increasing if argument `period` is not specified. Otherwise,
        `custom_datapoints` is internally sorted after normalizing the periodic boundaries
        with ``custom_datapoints = custom_datapoints % period``.
    :param tensorflow_datapoints: The TF datapoints (y-coordinates) of the data points, same
        length as `custom_datapoints`.
    :return: The interpolated values, same shape as custom_datapoints.
    :raises: ValueError – If custom_datapoints and tensorflow_datapoints have different length If
        custom_datapoints and tensorflow_datapoints not 1-D sequences If period == 0
    """
    if custom_datapoints is not None and tensorflow_datapoints is not None:
        if not all(isinstance(dps, (list, tuple)) for dps in (custom_datapoints, tensorflow_datapoints)):
            raise TypeError(
                f"custom_datapoints and tensorflow_datapoints must be a list or tuple.\n"
                f"Current datatype: \ncustom_datapoints is a {type(custom_datapoints).__name__}, "
                f"while tensorflow_datapoints is a {type(tensorflow_datapoints).__name__}")

        if len(custom_datapoints) != len(tensorflow_datapoints):
            raise ValueError(
                f"custom_datapoints and tensorflow_datapoints must be the same length. "
                f"custom_datapoints and tensorflow_datapoints have {len(custom_datapoints)} "
                f"and {len(tensorflow_datapoints)} values, respectively")

        if not isinstance(custom_value, (int, float)):
            return tuple(np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints))
        else:
            return np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints)

    else:
        return custom_value

def get_zoom_matrix(zooms, image_height, image_width, name=None):
    """Returns projective transform(s) for the given zoom(s).

    Args:
      zooms: A matrix of 2-element lists representing `[zx, zy]` to zoom for
        each image (for a batch of images).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.

    Returns:
      A tensor of shape `(num_images, 8)`. Projective transforms which can be
        given to operation `image_projective_transform_v2`.
        If one row of transforms is
         `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
         `(x, y)` to a transformed *input* point
         `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
         where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "zoom_matrix"):
        num_zooms = tf.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Zoom matrices are always float32.
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0:1])
        y_offset = ((image_height - 1.0) / 2.0) * (1.0 - zooms[:, 1:2])

        return tf.concat(
            values=[
                zooms[:, 0:1],
                tf.zeros((num_zooms, 1), tf.float32),
                x_offset,
                tf.zeros((num_zooms, 1), tf.float32),
                zooms[:, 1:2],
                y_offset,
                tf.zeros((num_zooms, 2), tf.float32),
            ],
            axis=1,
    )


@keras.saving.register_keras_serializable(package='Augmentation', name='AugmentationLayer')
class AugmentationLayer(base_layer.BaseRandomLayer):

    def __init__(self,
                 active: bool = True,
                 custom_interp_datapoints: Union[List, Tuple, None] = None,
                 tensorflow_interp_datapoints: Union[List, Tuple, None] = None,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'dual_augmentation_layer',
                 **kwargs):
        """
        Initializes the base augmentation layer for image, mask, and bounding box transformations.

        This layer serves as a foundation for implementing custom augmentation behaviors that
        may operate on different input types (images, targets, bounding boxes). It also supports
        optional interpolation between custom and TensorFlow-defined datapoints, useful for
        mapping custom augmentation scales to standardized outputs.

        :param active: If False, the augmentation logic is disabled (passthrough mode).
                       Stored as a `tf.Variable` to allow dynamic switching at runtime.
        :param custom_interp_datapoints: Optional list or tuple of reference x-values
                       for interpolation from custom-defined space.
        :param tensorflow_interp_datapoints: Optional list or tuple of reference y-values
                       for interpolation into the TensorFlow space.
        :param image_key: Dictionary key under which the image tensor is expected in the input.
        :param target_key: Dictionary key under which the target (e.g., mask) tensor is expected.
        :param bounding_box_key: Dictionary key under which the bounding box tensor is expected.
        :param batches_before_aug: int, optional.
            Number of batches to pass through the layer without augmentation at the start of each epoch.
            Augmentation is disabled for these batches to ensure the model sees the unaltered original data first.
            After this threshold, augmentation is applied for the remaining batches in the epoch.

        :param name: Name of the layer.
        :param kwargs: Additional keyword arguments passed to `BaseRandomLayer`.
        """
        super().__init__(trainable=False, name=name, **kwargs)

        self.check_parameter_type(params_and_types={
            "active": (active, bool, False),
            "custom_interp_datapoints": (custom_interp_datapoints, (list, tuple), True),
            "tensorflow_interp_datapoints": (tensorflow_interp_datapoints, (list, tuple), True),
            "image_key": (image_key, str, False),
            "target_key": (target_key, str, False),
            "bounding_box_key": (bounding_box_key, str, False),
            "batches_before_aug": (batches_before_aug, int, True),
            "name": (name, str, False),
        })

        self.active = tf.Variable(tf.constant(active, dtype=tf.bool))
        self.tensorflow_interp_datapoints = tensorflow_interp_datapoints
        self.custom_interp_datapoints = custom_interp_datapoints
        self.image_key = image_key
        self.target_key = target_key
        self.bounding_box_key = bounding_box_key
        self.target_compute_dtype = tf.int32

        if batches_before_aug is None:
            self.batches_before_aug = tf.constant(value=0, dtype=tf.int32)
        else:
            self.batches_before_aug = tf.constant(batches_before_aug, dtype=tf.int32, name='batches_before_aug')

        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.int32, name='batch_counter')

    def __repr__(self):
        return f"{self.name}"

    @doc_controls.for_subclass_implementers
    def augment_image(self, image, transformation):
        """Augment a batch of image(s) during training.

        Args:
          image: 4D image input tensor to the layer. Forwarded from
            `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def augment_label(self, label, transformation):
        """Augment a batch of label(s) during training.

        Args:
          label: 4D [B,H,W,C] label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_target(self, target, transformation):
        """Augment a batch of target(s) during training.

        Args:
          target: 4D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        return self.augment_label(target, transformation)

    @doc_controls.for_subclass_implementers
    def augment_bounding_boxes(self, images,
                               bounding_boxes,
                               transformation: Optional[Dict[str, Any]] = None):
        """Augment bounding boxes for one image during training.

        Args:
          images: 4D image input tensor to the layer. Forwarded from
            `layer.call()`.
          bounding_boxes: 3D Tensor [B, N, 4] of bounding boxes to the layer. Forwarded from
            `call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        """Produce random transformation config for batched input.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          image: 4D image tensor from inputs.
          label: optional 1D/3D/4D label tensor from inputs.
          bounding_box: optional 2D bounding boxes tensor from inputs.

        Returns:
          Any type of object, which will be forwarded to `augment_image`,
          `augment_label` and `augment_bounding_box` as the `transformation`
          parameter.
        """
        return None

    def check_inputs_type(self, inputs):
        """Ensures the input is a Python dictionary"""
        if not isinstance(inputs, dict):
            raise TypeError(f"The inputs must be a dict, with keys ['{self.image_key}', "
                            f"'{self.target_key}', {self.bounding_box_key}]. "
                            f"Got {inputs} of type {type(inputs).__name__}")

    @staticmethod
    def check_parameter_type(params_and_types: Dict[str, Tuple[Any, Union[Type, Tuple[Type, ...]], bool]]):
        """
        Validates that each parameter value matches its expected type, and handles optional parameters.

        :param params_and_types: A dictionary where each key is a parameter name, and the value is a tuple:
            (value, expected_type(s), is_optional), where:
                - value: The actual value to validate.
                - expected_type(s): A type or tuple of types that the value is expected to match.
                - is_optional: If True, allows the value to be None.

            Example:
                check_type({
                    "name": ("Alice", str, False),
                    "age": (None, int, True),
                    "scores": ([90, 95], list, False),
                })

        :raises TypeError:
            - If a value is None and is_optional is False.
            - If a value does not match the expected type(s).
        """
        for param_name, (value, expected_types, is_optional) in params_and_types.items():
            if value is None:
                if is_optional:
                    continue
                else:
                    raise TypeError(f"{param_name} must not be None.")

            if not isinstance(value, expected_types):
                allowed_types = (
                    expected_types.__name__
                    if isinstance(expected_types, type)
                    else ', '.join(t.__name__ for t in expected_types)
                )
                raise TypeError(f"{param_name} must be of type {allowed_types}. Got {type(value).__name__}.")

    def check_dict_keys(self, inputs):
        """Checks that the inputs have the correct dict keys"""
        if not all([key in inputs for key in [self.image_key, self.target_key]]):
            raise KeyError(f"Invalid dictionary keys for training examples. "
                           f"Got keys: {list(inputs.keys())}"
                           f"The Images and segmentation mask must be assigned to the "
                           f"keys '{self.image_key}' and '{self.target_key}', respectively. ")

    def _ensure_inputs_are_compute_dtype(self, inputs):
        """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
        self.check_inputs_type(inputs)
        self.check_dict_keys(inputs)
        inputs[self.image_key] = utils.ensure_tensor(inputs=inputs[self.image_key],
                                                     dtype=self.compute_dtype)

        inputs[self.target_key] = utils.ensure_tensor(inputs=inputs[self.target_key],
                                                      dtype=self.target_compute_dtype)
        return inputs

    @staticmethod
    def interpolate_to_tensorflow_value(custom_value: Union[int, float, list, tuple],
                                        custom_datapoints: Union[list, tuple, None],
                                        tensorflow_datapoints: Union[list, tuple, None]):
        """
        Produces TensorFlow equivalent of a given custom value. This is done by
        interpolating between custom values and their corresponding TensorFlow values.

        :param custom_value: The custom value to be converted Tensorflow equivalent.
        :param custom_datapoints: tuple or list, The   custom data points (x-coordinates).
            These values must be presented in ascending order.
        :param tensorflow_datapoints: The TF equivalent (y-coordinates) of the custom datapoints, same
            length as `custom_datapoints`.
        :return: The interpolated value (The Tensorflow equivalent)
        :raises: ValueError – If custom_datapoints and tensorflow_datapoints have different length If
            custom_datapoints and tensorflow_datapoints not 1-D sequences If period == 0
        """
        if custom_datapoints is not None and tensorflow_datapoints is not None:
            if not all(isinstance(dps, (list, tuple)) for dps in (custom_datapoints, tensorflow_datapoints)):
                raise TypeError(
                    f"custom_datapoints and tensorflow_datapoints must be a list or tuple.\n"
                    f"Current datatype: \ncustom_datapoints is a {type(custom_datapoints).__name__}, "
                    f"while tensorflow_datapoints is a {type(tensorflow_datapoints).__name__}")

            if len(custom_datapoints) != len(tensorflow_datapoints):
                raise ValueError(
                    f"custom_datapoints and tensorflow_datapoints must be the same length. "
                    f"custom_datapoints and tensorflow_datapoints have {len(custom_datapoints)} "
                    f"and {len(tensorflow_datapoints)} values, respectively")

            if not isinstance(custom_value, (int, float)):
                return tuple(np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints))
            else:
                return np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints)

        else:
            return custom_value

    @staticmethod
    def is_unbatched(tensor: tf.Tensor) -> bool:
        """
        Determines if the input tensor is unbatched (i.e., 3D).

        Args:
            tensor: A TensorFlow tensor, typically with shape [B, H, W, C] or [H, W, C].

        Returns:
            True if the tensor is unbatched (rank == 3), False otherwise.
        """
        return tensor.shape.rank == 3

    @staticmethod
    def is_batched(tensor: tf.Tensor) -> bool:
        """
        Determines if the input tensor is batched (i.e., 4D).

        Args:
            tensor: A TensorFlow tensor, typically with shape [B, H, W, C] or [H, W, C].

        Returns:
            True if the tensor is batched (rank == 4), False otherwise.
        """
        return tensor.shape.rank == 4

    def ensure_batched(self, tensor: tf.Tensor):
        """
        Ensures that the input tensor is batched (rank 4).

        If it's unbatched (rank 3), this method expands its dimensions.
        Also returns a boolean indicating whether the tensor was originally unbatched.

        Args:
            tensor: A TensorFlow tensor with rank 3 or 4.

        Returns:
            A tuple:
                - The tensor in batched form (rank 4)
                - A boolean flag `was_unbatched` indicating if batching was applied
        """
        was_unbatched = self.is_unbatched(tensor)
        if was_unbatched:
            tensor = tf.expand_dims(input=tensor, axis=0)
        return tensor, was_unbatched

    def reset_counter(self):
        self.batch_counter.assign(0)

    def call(self, inputs, training=None):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)

        if training:
            # Increment the batch counter and decide whether to apply augmentation
            # based on whether the model has already seen enough original (non-augmented) batches.
            current_batch = self.batch_counter.assign_add(1)

            def augment():
                return tf.cond(
                    pred=tf.convert_to_tensor(self.active),
                    true_fn=lambda: self._augment(inputs),
                    false_fn=lambda: inputs)

            def no_augment():
                return inputs

            return tf.cond(
                pred=tf.greater(current_batch, self.batches_before_aug),
                true_fn=augment,
                false_fn=no_augment
            )
        else:
            return inputs

    def _augment(self, inputs):
        """Augments the image, target and bounding box."""
        image = inputs.get(self.image_key, None)
        target = inputs.get(self.target_key, None)
        bounding_box = inputs.get(self.bounding_box_key, None)

        original_image_shape = image.shape
        # The transform op only accepts rank 4 inputs,
        # so if we have an unbatched image,
        # we need to temporarily expand dims to a batch.
        image, image_was_unbatched = self.ensure_batched(image)

        transformation = self.get_random_transformation(image=image, label=target,
                                                        bounding_box=bounding_box)

        image = self.augment_image(image, transformation=transformation)

        if image_was_unbatched:
            image = tf.squeeze(image, 0)
            image.set_shape(original_image_shape)

        result = {self.image_key: image}

        # apply target augmentation only if target exist, and it is at least 3-dimensional
        if target is not None:
            original_target_shape = target.shape
            target, target_was_unbatched = self.ensure_batched(target)

            if self.is_batched(target):
                target = self.augment_target(target, transformation=transformation)
                if target_was_unbatched:
                    target = tf.squeeze(input=target, axis=0)
                    target.set_shape(original_target_shape)

                target = tf.round(target)
                result[self.target_key] = tf.cast(target, dtype=self.target_compute_dtype)
            elif target.shape.rank > 2 and target.shape[-1] > 1:
                raise ValueError(f"The shape of the target(s) must be in the form "
                                 f"[Height, Width, 1], got {target.shape}.")
            else:
                # incase the target is not a 3-dimensional tensor(e.g. a categorical target)
                result[self.target_key] = target

        if bounding_box is not None:
            bounding_box = self.augment_bounding_boxes(image, bounding_box,
                                                       transformation=transformation)
            result[self.bounding_box_key] = bounding_box

        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'active': self.active,
                'custom_interp_datapoints': self.custom_interp_datapoints,
                'tensorflow_interp_datapoints': self.tensorflow_interp_datapoints,
                'image_key': self.image_key,
                'target_key': self.target_key,
                'bounding_box_key': self.bounding_box_key,
                'batches_before_aug': self.batches_before_aug
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomRotation')
class RandomRotation(AugmentationLayer):
    """
    A preprocessing layer which randomly rotates images during training.

    This layer applies random rotations to each image and its segmentation mask (if available),
    filling empty regions according to the specified `fill_mode` and `fill_value`. The rotation
    angle is sampled uniformly from a specified range. At inference time or when `active=False`,
    the layer behaves as a no-op, passing the input (image and label) through unchanged.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched) or 4D (batched) tensor}

        Image and target tensors must follow the shape `(..., height, width, channels)`
        using the `"channels_last"` format.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched) or 4D (batched) tensor}

        Output shape and format are identical to input. NOTE: if the shape of the target is 2D,
        it will not be augmented.

    Arguments:
      angle: A float, int, tuple, or list representing the angle by which the image and mask
         will be rotated. If a single value (int or float) is provided, the rotation range is
         between [-angle, angle]. If a tuple or list of two values is provided, the angle is
         sampled between (angle[0], angle[1]).
      active: Boolean indicating whether to apply augmentation during training. If False,
         inputs are returned unchanged.
      fill_mode: Points outside the boundaries of the input are filled according to this mode.
         One of: `{"constant", "reflect", "wrap", "nearest"}`.

         - *reflect*: `(d c b a | a b c d | d c b a)` — reflects about the edge of the last pixel.
         - *constant*: `(k k k k | a b c d | k k k k)` — fills with constant value `k` (default 0).
         - *wrap*: `(a b c d | a b c d | a b c d)` — wraps around to the opposite edge.
         - *nearest*: `(a a a a | a b c d | d d d d)` — extends using nearest edge pixel.
      interpolation: Interpolation mode used during resampling. Supported values: `"nearest"`, `"bilinear"`.
      fill_value: A float representing the pixel value used to fill empty regions when `fill_mode="constant"`.
      image_key: The key used to retrieve the image tensor from the input dictionary.
      target_key: The key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: The key used to retrieve bounding box data from the input dictionary.
      name: The name of the layer instance.

    Raises:
      TypeError: If `angle` is not a float, int, list, or tuple.
      ValueError: If `angle` is a list or tuple not of length 2, or if the upper angle is less than the lower.
    """

    def __init__(self,
                 angle: Union[int, float, List[int], Tuple[int, int]] = 180,
                 active: bool = True,
                 fill_mode="reflect",
                 interpolation="bilinear",
                 fill_value=0.0,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'random_rotation',
                 **kwargs):

        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)
        self.angle = angle
        if isinstance(angle, (tuple, list)):
            if len(angle) != 2:
                raise ValueError(f"The {type(angle).__name__} `angle` must have only two values, got {angle}")
            self.lower = np.deg2rad(angle[0])
            self.upper = np.deg2rad(angle[1])

        elif isinstance(angle, (int, float)):
            self.lower = np.deg2rad(-angle)
            self.upper = np.deg2rad(angle)
        else:
            raise TypeError(
                f"The angle must be a list, tuple, int or float. Got {angle} of type `{type(angle).__name__}`"
            )

        if self.upper < self.lower:
            raise ValueError(
                "The upper boundary cannot be less than the lower boundary. Got angle = {}".format(angle)
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        # min_angle = self.lower * 2.0 * np.pi
        # max_angle = self.upper * 2.0 * np.pi
        batch_size = tf.shape(image)[0]
        angles = tf.random.uniform(shape=[batch_size], minval=self.lower, maxval=self.upper)

        # # 50% chance to apply rotation: mask = 1 if apply, 0 if skip
        # apply_mask = tf.cast(tf.random.uniform(shape=[batch_size], minval=0, maxval=1) < 0.5,
        #                      dtype=angles.dtype)
        #
        # # Zero out angles for samples where augmentation should be skipped
        # angles = angles * apply_mask

        return {"angles": angles}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        image_shape = tf.shape(image)

        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        angle = transformation["angles"]

        output = transform(images=image,
                           transforms=get_rotation_matrix(angle, img_hd, img_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation=self.interpolation,
                           )
        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        image_shape = tf.shape(label)

        label_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        label_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        angles = transformation["angles"]
        output = transform(label,
                           get_rotation_matrix(angles, label_hd, label_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation='nearest',
                           )
        return output

    def augment_bounding_boxes(self, images, bounding_boxes, transformation):
        """
        Applies rotation to bounding boxes for a batch of images.

        :param images: 4D tensor of shape (B, H, W, C)
        :param bounding_boxes: RaggedTensor of shape (B, None, 4), each row is [x_min, y_min, x_max, y_max]
        :param transformation: Dictionary containing 'angles': 1D tensor of shape (B,)
        :return: RaggedTensor of shape (B, None, 4) with updated bounding boxes
        """
        angles = -transformation["angles"]  # Clockwise for TF transform
        batch_size = tf.shape(images)[0]
        image_shapes = tf.shape(images)

        def rotate_boxes_per_image(args):
            boxes, angle, img = args
            h = tf.cast(tf.shape(img)[H_AXIS], tf.float32)
            w = tf.cast(tf.shape(img)[W_AXIS], tf.float32)
            cx, cy = w / 2.0, h / 2.0

            x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
            corners = tf.concat([
                tf.stack([x1, y1], axis=2),
                tf.stack([x2, y1], axis=2),
                tf.stack([x2, y2], axis=2),
                tf.stack([x1, y2], axis=2)
            ], axis=1)  # shape (N, 4, 2)

            translated = corners - tf.constant([[cx, cy]], dtype=tf.float32)

            cos_theta = tf.cos(angle)
            sin_theta = tf.sin(angle)
            rot_matrix = tf.convert_to_tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=tf.float32)

            rotated = tf.einsum('nij,jk->nik', translated, rot_matrix) + tf.constant([[cx, cy]], dtype=tf.float32)

            min_coords = tf.reduce_min(rotated, axis=1)
            max_coords = tf.reduce_max(rotated, axis=1)
            return tf.concat([min_coords, max_coords], axis=1)

        batched_outputs = tf.map_fn(
            fn=rotate_boxes_per_image,
            elems=(bounding_boxes, angles, images),
            fn_output_signature=tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
        )

        return batched_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"angle": self.angle,
                       "fill_mode": self.fill_mode,
                       "interpolation": self.interpolation,
                       "fill_value": self.fill_value})

        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomGaussianNoise')
class RandomGaussianNoise(AugmentationLayer):
    """
    A preprocessing layer which randomly applies Gaussian noise to images during training.

    This layer adds pixel-wise Gaussian noise to the input image tensor. The standard deviation
    is sampled uniformly from a specified range. Noise is applied only during training when
    `active=True`. During inference or when inactive, the layer behaves as a no-op: all inputs,
    including images, targets, and bounding boxes, are returned unmodified.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Image tensors must follow the shape `(..., height, width, channels)` using
        the `"channels_last"` format. Targets can be any of the following:
        - 1D: single class label
        - 2D: single-channel binary mask
        - 3D: multi-channel mask or label image
        - 4D: batched masks or labels

        Only the image is augmented; the target is passed through unchanged.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Output shape and format are identical to input.

    Arguments:
      stddev: A float, list, or tuple representing the lower and upper bounds of the
          standard deviation to use for sampling Gaussian noise. If a single float is provided,
          it is treated as the upper bound with the lower bound defaulting to 0.0. Both values
          must be ≥ 0.
      name: The name assigned to the layer instance.
      active: Boolean indicating whether this layer is active during training. If False,
          the input is returned unchanged.
      image_key: The key used to retrieve the image tensor from the input dictionary.
      target_key: The key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: The key used to retrieve bounding box data from the input dictionary.
      kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
      TypeError: If `stddev` is not a float, int, list, or tuple.
      ValueError: If `stddev` contains negative values or if the upper bound is less than the lower.
    """

    def __init__(self,
                 stddev: Union[float, List[float], Tuple[float, float]] = (0.0, 3.0),
                 name='random_gaussian_noise',
                 active: bool = True,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 **kwargs):

        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomDualGaussianNoise").set(True)

        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)
        self.stddev = stddev

        if isinstance(stddev, (tuple, list)):
            self.lower = stddev[0]
            self.upper = stddev[1]
        elif isinstance(stddev, (int, float)):
            self.lower = 0.0
            self.upper = stddev
        else:
            raise TypeError(
                "Allowed datatypes  for stddev: tuple, list, int, "
                "float. Receive {} of type - {}.".format(stddev, type(stddev).__name__))

        if self.lower < 0 or self.upper < 0:
            raise ValueError("stddev must be greater than 0, got {}".format(stddev))

        if self.upper < self.lower:
            raise ValueError(
                "The upper limit of stddev must be greater than the lower limit. "
                "{} is less than {}".format(self.upper, self.lower))

    def get_random_transformation(self, image=None, label=None, bounding_box=None):

        min_stddev = self.lower
        max_stddev = self.upper
        batch_size = tf.shape(image)[0]
        stddev = tf.random.uniform(shape=[batch_size], minval=min_stddev, maxval=max_stddev)

        # broadcast stddev to [batch_size, 1, 1, 1], the noise added to each image
        # in the batch uses a different standard deviation.
        stddev = tf.reshape(stddev, [-1, 1, 1, 1])
        return {"stddev": stddev}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        stddev = transformation["stddev"]

        output = image + tf.random.normal(shape=tf.shape(image),
                                          mean=0.0,
                                          stddev=stddev,
                                          dtype=image.dtype, )

        return output

    def augment_label(self, label, transformation):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev, })
        return config