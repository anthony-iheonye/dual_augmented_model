import einops
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from skimage.measure import label
from skimage.morphology import (binary_erosion, binary_dilation, disk, binary_opening,
                                remove_small_objects, remove_small_holes)
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='Unet', name='MergeLabelAxis')
class MergeLabelAxis(Layer):
    """Creates a 2D mask by combining the label channels that make up the predicted mask"""

    def __init__(self, name='mask_2D'):
        super().__init__(name=name, trainable=False, dtype=tf.int32)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, **kwargs):
        x = tf.argmax(input=inputs, axis=-1, output_type=self.dtype)
        return x


@register_keras_serializable(package='Unet', name='MorphologicalLayer')
class MorphologicalLayer(Layer):
    def __init__(self, opening_radius: int = 5, closing_radius: int = 3, erosion_radius: int = 2,
                 dilation_radius: int = 1, name='morphology_layer', min_size: int = 2000,
                 area_threshold: int = 300, **kwargs):
        """
        Applies series of morphological processing to the 2D version of the predicted mask.

        :param opening_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological opening on the mask.
        :param closing_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological closing on the mask.
        :param erosion_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological erosion on the mask.
        :param dilation_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological dilation on the mask.
        :param name: (str) name of the layer.
        """
        super(MorphologicalLayer, self).__init__(name=name, trainable=False, dtype=tf.float32, **kwargs)
        self.min_size = min_size
        self.area_threshold = area_threshold
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.erosion_radius = erosion_radius
        self.dilation_radius = dilation_radius
        self.opening_disk = disk(self.opening_radius)
        self.closing_disk = disk(self.closing_radius)
        self.erosion_disk = disk(self.erosion_radius)
        self.dilation_disk = disk(self.dilation_radius)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs):
        processed_mask = tf.map_fn(fn=self._tf_apply_morphological_operations, elems=inputs)
        return processed_mask

    def _apply_morphological_operations(self, mask):
        """Apply morphological operations to the 2D-mask."""
        processed_mask = label(label_image=mask, connectivity=1)

        # The remove_small_object should only be applied to mask containing more than one pea. If they contain one or
        # none, the function would not be applied. Instead we convert mask to a boolean, since the subsequent function
        # remove_small_holes can be applied to a boolean mask.
        if processed_mask.max() > 1:
            processed_mask = remove_small_objects(processed_mask, min_size=self.min_size).astype(np.bool_)
        else:
            processed_mask = processed_mask.astype(np.bool_)

        processed_mask = remove_small_holes(processed_mask, area_threshold=self.area_threshold, connectivity=1)
        processed_mask = binary_erosion(processed_mask, self.erosion_disk)
        processed_mask = binary_opening(processed_mask, self.opening_disk)
        processed_mask = label(label_image=processed_mask, connectivity=1)

        if processed_mask.max() > 1:
            processed_mask = remove_small_objects(processed_mask, min_size=self.min_size)
        else:
            processed_mask = processed_mask.astype(np.bool_)

        processed_mask = binary_dilation(processed_mask, self.dilation_disk)
        return processed_mask

    @tf.function(experimental_relax_shapes=True)
    def _tf_apply_morphological_operations(self, mask):
        mask_shape = mask.shape
        [processed_mask, ] = tf.py_function(func=self._apply_morphological_operations,
                                            inp=[mask], Tout=[tf.int32])

        processed_mask.set_shape(shape=mask_shape)
        return processed_mask

    def get_config(self):
        config = super(MorphologicalLayer, self).get_config()
        config.update({"opening_radius": self.opening_radius,
                       "closing_radius": self.closing_radius,
                       "erosion_radius": self.erosion_radius,
                       "dilation_radius": self.dilation_radius,
                       "min_size": self.min_size,
                       "area_threshold": self.area_threshold})
        return config


@register_keras_serializable(package='Unet', name='ExpandLabelAxis')
class ExpandLabelAxis(Layer):
    def __init__(self, num_classes: int, name='processed_mask'):
        super(ExpandLabelAxis, self).__init__(name=name, trainable=False, dtype=tf.int32)
        self.num_classes = num_classes

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, **kwargs):
        x = self._add_label_channels(mask=inputs)
        return x

    def _add_label_channels_old(self, mask):
        """Adds label channels to the mask."""
        if mask.shape[-1] > 1:
            return mask

        mask = tf.cast(mask, dtype=tf.int32)
        stack_array = tf.TensorArray(dtype=tf.float64, size=self.num_classes, dynamic_size=False)

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. If we have three classes - background, pea, and outline,
        # we would have a final mask whose last axis will have 3 layers/channels.
        for class_index in tf.range(self.num_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the integer 'class_index'.
            temp_mask = tf.equal(mask[..., 0], tf.constant(class_index, dtype=tf.int32))
            # add each temporary mask to the TensorArray.
            stack_array = stack_array.write(index=class_index,
                                            value=tf.cast(temp_mask, dtype=tf.float64))

        # stack all the temporary masks within the stack_array, so together they form the third axis of the
        # overall mask. The result have the shape [num_classes, height, width], hence we will reshape using
        # the `einops.rearrange` method
        mask = stack_array.stack()
        mask = einops.rearrange(tensor=mask,
                                pattern='num_classes batch height width -> batch height width num_classes')
        return mask

    def _add_label_channels(self, mask):
        """Adds label channels to the mask."""
        if mask.shape[-1] > 1:
            return mask
        mask = tf.cast(mask, dtype=tf.int32)

        stack_list = []

        # Create individual tensor array for each class. Each tensor would display the area on the original
        # mask that is occupied by individual classes. if we have three class - background, pea, and outline,
        # we would have three tensor arrays.
        for intensity in tf.range(self.num_classes):
            # Produce a temporary mask depicting all the pixel locations on the original tensor named 'mask'
            # that have the same pixel intensity as  the integer 'class_index'. we want to
            temp_mask = tf.equal(mask[..., 0], tf.constant(intensity, dtype=tf.int32))
            # add each temporary mask to the stack_list.
            stack_list.append(tf.cast(temp_mask, dtype=tf.float32))

        # stack all the temporary masks within the stack_list, so together they form the third axis of the
        # overall mask. Hence, the overall mask would be of dimension [height, width, number_of_classes]
        mask = tf.stack(stack_list, axis=-1)  # Axis starts from 0, so axis of 2 represents the third axis
        return mask


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

