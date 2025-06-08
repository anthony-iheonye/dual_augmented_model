from typing import Optional, Tuple

from keras.applications import Xception
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='VisualAttribute', name='XceptionBackbone')
class XceptionBackbone(Layer):
    """Creates the Xception backbone for the tf.keras model class - `VisualAttribute`."""

    def __init__(self,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 initialize_with_imagenet: bool = False,
                 name='xception', **kwargs):
        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.initialize_with_imagenet_weight = initialize_with_imagenet
        self.backbone = Xception(include_top=False,
                                 weights='imagenet' if initialize_with_imagenet else None,
                                 input_shape=image_shape)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        x = self.backbone(inputs)
        return x

    @property
    def trainable_layers(self):
        return self.backbone.layers

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_shape': self.image_shape,
            'name': self.name,
            'initialize_with_imagenet': self.initialize_with_imagenet_weight
        })
        return config