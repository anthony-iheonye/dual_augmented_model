from typing import List
from typing import Tuple, Union, Optional

import keras
import tensorflow as tf

from app.layers.augmentation import (RandomAugmentationLayer, PassThroughLayer, NormalizeImage, SplitImageTarget)
from app.layers.segmenter import ExpandLabelAxis


@keras.saving.register_keras_serializable(package='Augmentation', name='AugmentedModel')
class AugmentedModel(tf.keras.Model):
    """
    A Keras model wrapper that integrates real-time image-target augmentation directly into the training pipeline.

    Designed for tasks like semantic segmentation, this model applies GPU-accelerated augmentations
    to image-mask (and optionally bounding box) pairs using a unified interface. It wraps a
    user-defined base model, applying augmentation only during training.

    To apply augmentation on the targets, the targets must either be 3D or 4D tensors (i.e., shape `[H, W, C]` or `[B, H, W, C]`). If `targets` are scalar (e.g., shape `[B]`), they will
    be passed through unchanged.

    Supported Transformations:
        - Horizontal Flip
        - Vertical Flip
        - Rotation
        - Shear
        - Zoom
        - Gaussian Noise
        - Brightness Adjustment
        - Contrast Adjustment

    Each augmentation can be selectively enabled via constructor arguments.

    Example:
        >>> model = AugmentedModel(
        >>>     num_of_classes=3,
        >>>     base_model=unet_model,
        >>>     flip=True,
        >>>     flip_mode="HORIZONTAL_AND_VERTICAL",
        >>>     rotate=True,
        >>>     theta=(-20, 20),
        >>>     shear=True,
        >>>     shear_angle=(-5, 5),
        >>>     zoom=True,
        >>>     zoom_factor=(90, 110),
        >>>     use_gaussian_noise=True,
        >>>     gaussian_range=(0.0, 0.02),
        >>>     adjust_brightness=True,
        >>>     brightness_range=(0.85, 1.15),
        >>>     normalize=True
        >>> )

    :param num_of_classes: (int): Number of classes on the 3D/4D targets. This information will be
        used for one-hot or channel-wise label expansion.
    :param base_model: (tf.keras.Model, optional): The user-defined core model that will be trained
        with augmented data produced within model during training.
    :param image_key: (str): Key used to extract image tensors from the dictionary-based tf input
        Dataset.
    :param target_key: (str): Key used to extract target tensors from the dictionary-based tf input
        Dataset.
    :param bounding_box_key: (str): Key used to extract bounding boxes, if available.
    :param batches_before_aug: (Optional[int]): Number of batches input data to pass through the
        model (per epoch), before initiating augmentation. For instance, is to be trained with 200 batches of images/masks per epoch (steps_per_epoch = 250), and the batches_before_aug is set
        to 50, at the start of each epoch, the first 50 batches will be not be augmented, while the remaining 150 batches will be augmented. This technique ensures the model learns patterns from all the original and augmented data.
    :param flip: (bool): Enable random flipping. Default: False.
    :param flip_mode: (str): Flip direction. Options:
            - 'HORIZONTAL': left-right flip
            - 'VERTICAL': up-down flip
            - 'HORIZONTAL_AND_VERTICAL': both directions randomly

    :param rotate: (bool): Enable random rotation. Default: False.
    :param theta: (int or Tuple[int, int]): Rotation angle in degrees. e.g. (-30, 30).
            - If int: maximum absolute angle. e.g., 30 → (-30°, +30°)
            - If tuple: explicit range (e.g., (-45, 45))

    :param shear: (bool): Enable shearing. Default: False.
    :param shear_angle: (float or Tuple[float, float]): Shear angles in degrees. eg.: (-10.0, 10.0).

    :param zoom: (bool): Enable zooming. Default: False.
    :param zoom_factor: (int or Tuple[int, int]): Zoom in/out as percentage. eg. (80, 120). Setting zoom factor to
            -100 = no zoom. (90, 110) → 0.9x–1.1x zoom range

    :param use_gaussian_noise: (bool): Add Gaussian noise to input. Default: False.
    :param gaussian_range: Tuple of floats defining standard deviation of noise.
                           E.g., `(0.0, 0.02)` where 0.02 = 2% noise. Must be ≥ 0.

    :param adjust_brightness: (bool): Adjust brightness randomly. Default: False.
    :param brightness_range: Tuple of floats. 1.0 means no change.
                             E.g., `(0.9, 1.1)` varies brightness ±10%.

    :param normalize: (bool): Whether to normalize images before inference. Default: True.
    :param scale_to_0_1: (bool): Whether to scale pixel values to [0, 1]. Default: True. If it is set to False, the pixel will be scaled to [-1, 1].
    :param fill_mode: (str): Pixel fill mode for rotation/shear/zoom. Options: 'constant', 'reflect', 'nearest'. Default: 'reflect'.
    :param interpolation: Image resampling method: `"nearest"` or `"bilinear"`. Segmentation targets always use `"nearest"`. Images are usually set to bilinear.
    :param fill_value: (float): Value to use when `fill_mode='constant'`. Default: 0.0.
    :param image_intensity_range: (Tuple[float, float]): Min/max pixel range. Default: (0.0, 255.0)
    :param name: (str) Name of the model wrapper instance.

    Returns:
        Tuple (y_pred, target) during training;
        Single prediction tensor during inference.
    """

    def __init__(self,
                 num_of_classes: int,
                 base_model: tf.keras.Model = None,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = 'bounding_boxes',
                 batches_before_aug: Optional[int] = None,
                 flip: bool = False,
                 flip_mode: str = 'HORIZONTAL_AND_VERTICAL',
                 rotate: bool = False,
                 theta: Union[int, Tuple[int, int], None] = 180,
                 shear: bool = False,
                 shear_angle: Union[list, tuple, float, None] = (0, 2),
                 zoom: bool = False,
                 zoom_factor: Union[int, list, tuple, None] = (90, 110),
                 use_gaussian_noise: bool = False,
                 gaussian_range: Union[float, Tuple[float, float]] = (0.0, 0.01),
                 adjust_brightness: bool = False,
                 brightness_range: Union[float, list, tuple] = (0.90, 1.10),
                 jitter: bool = False,
                 jitter_range: Union[List[int], Tuple[int, int]] = (0, 60),
                 normalize: bool = True,
                 scale_to_0_1: bool = True,
                 fill_mode: str = "reflect",
                 interpolation: str = "nearest",
                 fill_value: float = 0.0,
                 image_intensity_range: Tuple[float, float] = (0.0, 255.0),
                 name: str = 'augmented_model',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.base_model = base_model if base_model is not None else PassThroughLayer()
        self.image_key = image_key
        self.target_key = target_key
        self.bounding_box_key = bounding_box_key
        self.batches_before_aug = batches_before_aug

        self.num_of_classes = num_of_classes
        self.flip = flip
        self.flip_mode = flip_mode
        self.rotate = rotate
        self.theta = theta
        self.shear = shear
        self.shear_angle = shear_angle
        self.zoom = zoom
        self.zoom_factor = zoom_factor
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_range = gaussian_range
        self.adjust_brightness = adjust_brightness
        self.brightness_range = brightness_range
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.normalize = normalize
        self.scale_to_0_1 = scale_to_0_1
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.image_intensity_range = image_intensity_range

        self.augmenter = RandomAugmentationLayer(image_key=image_key,
                                                 target_key=target_key,
                                                 bounding_box_key=bounding_box_key,
                                                 batches_before_aug=batches_before_aug,
                                                 flip=flip,
                                                 flip_mode=flip_mode,
                                                 rotate=rotate,
                                                 theta=theta,
                                                 shear=shear,
                                                 shear_angle=shear_angle,
                                                 zoom=zoom,
                                                 zoom_factor=zoom_factor,
                                                 use_gaussian_noise=use_gaussian_noise,
                                                 gaussian_range=gaussian_range,
                                                 adjust_brightness=adjust_brightness,
                                                 brightness_range=brightness_range,
                                                 jitter=jitter,
                                                 jitter_range=jitter_range,
                                                 fill_mode=fill_mode,
                                                 interpolation=interpolation,
                                                 image_intensity_range=image_intensity_range,
                                                 fill_value=fill_value,
                                                 name='random_augmentation_layer')

        self.split_image_target = SplitImageTarget(image_key=image_key,
                                                   target_key=target_key)

        self.expand_label_axis = ExpandLabelAxis(num_classes=num_of_classes)

        if normalize:
            self.normalizer = NormalizeImage(active=normalize, scale_to_0_1=scale_to_0_1)
        else:
            self.normalizer = PassThroughLayer()

    def call(self, inputs, training=None, mask=None):

        if training:
            image_target_dict = self.augmenter(inputs, training=training)
            # image_target_dict = self.normalizer(image_target_dict)

            # Extract images and targets batch from dictionary
            images, targets = self.split_image_target(image_target_dict)

            # Assign each target class to a separate channel. If the target
            # is not 3D or 4D, it will be returned untorched.
            targets = self.expand_label_axis(targets)

            # Forward pass
            y_pred = self.base_model(images)

            return y_pred, targets
        else:
            output = self.normalizer(inputs)
            output = self.base_model(output)
            return output

    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """

        x, y, sample_weight = self.unpack_x_y_sample_weight(data)
        inputs = self.combine_images_targets(x, y)

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred, target = self(inputs=inputs, training=True)
            loss = self.compute_loss(x, target, y_pred, sample_weight)

        self._validate_target_and_loss(y=target, loss=loss)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, target, y_pred, sample_weight)

    def combine_images_targets(self, x, y):
        """
        Returns a dictionary containing the images (input features) and
        targets

        :param x: dict or Tensor, A batch of images (features)
        :param y: dict or Tensor, A batch of targets (segmentation masks,
            categorical labels, bounding boxes). To include segmentation mask and
            bounding boxes, y  must be a dictionary, with keys - 'targets'
            and optionally (`bounding_boxes`), for segmentation masks and bounding boxes,
            respectively.
        """
        if all([not tf.is_tensor(x), not isinstance(x, dict)]):
            raise TypeError(
                f"Incorrect feature type. The image(s) must be a Tensor or a dict "
                f"with key - '{self.image_key}'. Got {x} of type `{type(x).__name__}`.")

        if all([not tf.is_tensor(y), not isinstance(y, dict)]):
            raise TypeError(
                f"Incorrect target type. The target(s) must be a Tensor or a dictionary "
                f"containing a batch of target(s), got {y} of type `{type(y).__name__}`. "
                f"If the target is a Tensor, it must be the segmentation mask. "
                f"To include bounding boxes, the target must be a dictionary. "
                f"In this case, the keys must be `{self.target_key}` and `{self.bounding_box_key}`."
                f"The target can also be a dictionary containing only the segmentation masks. "
                f"In that case,the key will be '{self.target_key}'")

        if isinstance(x, dict) and isinstance(y, dict):
            output = {self.image_key: x[self.image_key]}
            targets = y.get(self.target_key)
            b_boxes = y.get(self.bounding_box_key)

            if targets is not None:
                output[self.target_key] = targets
            if b_boxes is not None:
                output[self.bounding_box_key] = b_boxes
            return output

        elif isinstance(x, dict) and not isinstance(y, dict):
            return {self.image_key: x[self.image_key], self.target_key: y}

        elif not isinstance(x, dict) and isinstance(y, dict):
            output = {self.image_key: x}
            targets = y.get(self.target_key)
            b_boxes = y.get(self.bounding_box_key)

            if targets is not None:
                output[self.target_key] = targets
            if b_boxes is not None:
                output[self.bounding_box_key] = b_boxes

            return output

        else:
            return {self.image_key: x, self.target_key: y}

    def reset_counter(self):
        self.augmenter.reset_counter()

    def unpack_x_y_sample_weight(self, data):
        """Unpacks user-provided data tuple.

        This is a convenience utility to be used when overriding
        `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
        This utility makes it easy to support data of the form `(x,)`,
        `(x, y)`, or `(x, y, sample_weight)`.

        Standalone usage:

        >>> features_batch = tf.ones((10, 5))
        >>> labels_batch = tf.zeros((10, 5))
        >>> data = (features_batch, labels_batch)
        >>> # `y` and `sample_weight` will default to `None` if not provided.
        >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        >>> sample_weight is None
        True

        Example in overridden `Model.train_step`:

        ```python
        class MyModel(tf.keras.Model):

          def train_step(self, data):
            # If `sample_weight` is not provided, all samples will be weighted
            # equally.
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

            with tf.GradientTape() as tape:
              y_pred = self(x, training=True)
              loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
              trainable_variables = self.trainable_variables
              gradients = tape.gradient(loss, trainable_variables)
              self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            self.compiled_metrics.update_state(y, y_pred, sample_weight)
            return {m.name: m.result() for m in self.metrics}
        ```

        Args:
          data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

        Returns:
          The unpacked tuple, with `None`s for `y` and `sample_weight` if they are
          not provided.
        """
        if isinstance(data, list):
            data = tuple(data)

        if isinstance(data, dict):
            image = data.get(self.image_key)
            target = data.get(self.target_key)
            sample_weight = data.get('sample_weight')
            return image, target, sample_weight

        if not isinstance(data, tuple):
            return data, None, None
        elif len(data) == 1:
            return data[0], None, None
        elif len(data) == 2:
            return data[0], data[1], None
        elif len(data) == 3:
            return data[0], data[1], data[2]
        else:
            error_msg = (
                "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                "or `(x, y, sample_weight)`, found: {}"
            ).format(data)
            raise ValueError(error_msg)

