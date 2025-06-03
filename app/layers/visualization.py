import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

from .morphology import MorphologicalLayer, MergeLabelAxis, ExpandLabelAxis


@register_keras_serializable(package='Unet', name='InfocusPeas')
class InfocusPeas(Layer):
    """
    Produces an RGB image made of infocus peas. Input to this layer is a list containing the
    3D-mask and the original rgb image.
    """

    def __init__(self, name='infocus_peas'):
        super().__init__(name=name, trainable=False, dtype=tf.float32)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: a list containing the 3D-mask and the original rgb image.
        """
        mask_2d, rgb_image = inputs
        if len(mask_2d.shape) < 4:
            mask_2d = tf.expand_dims(mask_2d, axis=-1)
        x = tf.multiply(x=rgb_image, y=tf.cast(mask_2d, tf.float32))
        return x


@register_keras_serializable(package='Unet', name='PostProcessMask')
class PostProcessMask(Layer):
    """
    Applies morphological operations on the predicted mask, in order to produce a mask that is much closer to the
    ground truth mask. The output of this layer is a list containing the mask and the colored infocus peas.
    On the other hand, the inputs to the layer must be a list containing the decoder's output and the original rgb
    image, i.e. [decoder_output, image_input].
    """
    def __init__(self,
                 num_classes: int,
                 opening_radius: int = 5,
                 closing_radius: int = 3,
                 erosion_radius: int = 2,
                 dilation_radius: int = 1,
                 min_size: int = 2000,
                 area_threshold: int = 300,
                 name: str = 'post_process',
                 **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.num_classes = num_classes

        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.erosion_radius = erosion_radius
        self.dilation_radius = dilation_radius
        self.min_size = min_size
        self.area_threshold = area_threshold

        self.merge_label_axis = MergeLabelAxis(name='merged_mask')
        self.morphological_layer = MorphologicalLayer(opening_radius=opening_radius,
                                                      closing_radius=closing_radius,
                                                      erosion_radius=erosion_radius,
                                                      dilation_radius=dilation_radius,
                                                      min_size=min_size,
                                                      area_threshold=area_threshold,
                                                      name='processed_2D_mask')

        self.expand_label_axis = ExpandLabelAxis(num_classes=num_classes, name='processed_3D_mask')
        self.infocus_peas = InfocusPeas(name='infocus_peas')

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: The predicted mask from UNet, before morphological operations were applied.
        :return: The mask after minor morphological steps have been applied.
        """
        # Convert the 3D predicted_mask to a 2D mask, by collapsing the label axis.
        merged_mask = self.merge_label_axis(inputs)

        # Apply morphological operations
        processed_2d_mask = self.morphological_layer(merged_mask)

        # convert the morphed mask to 3D mask. Each channel represent a unique class
        processed_3d_mask = self.expand_label_axis(merged_mask)

        return processed_2d_mask, processed_3d_mask

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'opening_radius': self.opening_radius,
                       'closing_radius': self.closing_radius,
                       "erosion_radius": self.erosion_radius,
                       "dilation_radius": self.dilation_radius,
                       "min_size": self.min_size,
                       "area_threshold": self.area_threshold})
        return config
