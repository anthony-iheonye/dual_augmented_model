import keras
import tensorflow as tf
from keras.src.layers.preprocessing import preprocessing_utils as utils
from tensorflow.keras.layers import Layer
from .image_target_processing import AugmentationLayer


@keras.saving.register_keras_serializable(package='Augmentation', name='SplitImageTarget')
class SplitImageTarget(tf.keras.layers.Layer):
    """
    A utility Keras layer that extracts and returns image and target tensors from a dictionary input.

    This layer is useful in preprocessing pipelines where inputs are passed as dictionaries
    (e.g., from tf.data or custom augmentation layers), and models expect separate (image, target) tuples.

    Example:
        input = {'images': <image_tensor>, 'targets': <mask_tensor>}
        output = (<image_tensor>, <mask_tensor>)
    """

    def __init__(self,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 name='splitImageMask',
                 **kwargs):
        """
        Initializes the SplitImageTarget layer.

        :param image_key: str, the key used to retrieve the image tensor from the input dictionary.
        :param target_key: str, the key used to retrieve the target (e.g., mask) tensor from the input dictionary.
        :param name: str, name of the layer.
        :param kwargs: Additional keyword arguments passed to the base Layer class.
        """
        super(SplitImageTarget, self).__init__(name=name, trainable=False, **kwargs)
        self.image_key = image_key
        self.target_key = target_key

    def call(self, inputs, **kwargs):
        """
        Splits a dictionary input into a tuple of (image, target) tensors.

        :param inputs: dict, a dictionary containing image and target tensors.
                       Expected to have keys matching `self.image_key` and `self.target_key`.
        :return: Tuple (image_tensor, target_tensor)
        """
        return inputs[self.image_key], inputs[self.target_key]

    def get_config(self):
        config = super().get_config()
        config.update({'image_key': self.image_key,
                       'target_key': self.target_key})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='PassThroughLayer')
class PassThroughLayer(Layer):
    """Passes"""

    def __init__(self, name='pass_through', **kwargs):
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'name': self.name})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='NormalizeImage')
class NormalizeImage(AugmentationLayer):
    """
    A preprocessing layer that normalizes input images to a standardized float range.

    This layer scales image pixel values either to the range [0, 1] or [-1, 1] depending
    on the configuration. The normalization is only applied during training if `active=True`.

    Inputs:
        A dictionary with at least the key `'images'`, pointing to a 3D or 4D tensor
        representing the input image(s). The pixel values can be integers or floats.

    Outputs:
        The dictionary with the `'images'` tensor normalized.
        Other keys (e.g., `'targets'`) are returned unmodified.

    Example:
    >>> layer = NormalizeImage(scale_to_0_1=True)
    >>> inputs = {'images': tf.random.uniform((256, 256, 3), maxval=255, dtype=tf.int32)}
    >>> outputs = layer(inputs, training=True)
    """

    def __init__(self,
                 image_key: str = 'images',
                 active: bool = True,
                 scale_to_0_1: bool = True,
                 name: str = 'normalize_layer'):
        """
        Initializes the normalization layer.

        Args:
            active (bool): If False, the layer returns the inputs unchanged.
            scale_to_0_1 (bool): If True, normalize to [0, 1] using division by 255.
                                 If False, normalize to [-1, 1] using (x - 127.5) / 127.5.
            name (str): Name of the layer.
        """
        super().__init__(name=name, active=active, image_key=image_key)
        self.scale_to_0_1 = scale_to_0_1

        if self.scale_to_0_1:
            self.norm_fn = lambda image: tf.divide(x=tf.cast(image, tf.float32), y=255.0)
        else:
            self.norm_fn = lambda image: tf.divide(x=tf.cast(image, tf.float32) - 127.5, y=127.5)

    def __repr__(self):
        """Returns the name of the layer as its string representation."""
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        Applies normalization during training if `active` is True.

        Args:
            inputs (dict): Dictionary containing at least the key `'images'`.
            training (bool, optional): Whether the layer is in training mode.

        Returns:
            dict: The inputs with the `'images'` tensor normalized.
        """
        inputs = self._ensure_inputs_are_compute_dtype(inputs)

        return tf.cond(
            pred=tf.convert_to_tensor(self.active),
            true_fn=lambda: self.norm_fn(inputs),
            false_fn=lambda: inputs)

    def check_inputs_type(self, inputs):
        """Ensures the input is a Python dictionary"""
        if not isinstance(inputs, (dict, tf.Tensor)):
            raise TypeError(f"The inputs must be a tf.Tensor or a dict, with "
                            f"key '{self.image_key}'. "
                            f"Got {inputs} of type {type(inputs).__name__}")

    def check_dict_keys(self, inputs):
        """Checks that the inputs have the correct dict keys"""
        if not self.image_key in inputs:
            raise KeyError(f"Invalid dictionary for inputs. "
                           f"Got keys: {list(inputs.keys())}"
                           f"The Images  must be assigned to the "
                           f"key '{self.image_key}'. ")

    def _ensure_inputs_are_compute_dtype(self, inputs):
        """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
        self.check_inputs_type(inputs)

        if isinstance(inputs, dict):
            self.check_dict_keys(inputs)
            image = utils.ensure_tensor(inputs=inputs[self.image_key],
                                        dtype=self.compute_dtype)
            return image
        else:
            image = utils.ensure_tensor(inputs=inputs, dtype=self.compute_dtype)
            return image

    def get_config(self):
        """
        Returns the config dictionary for serialization.

        Returns:
            dict: Configuration of the layer.
        """
        config = super().get_config()
        config.update({'scale_to_0_1': self.scale_to_0_1})
        return config




