from typing import Optional, Tuple, List

from tensorflow.keras import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D)
from tensorflow.keras.saving import register_keras_serializable

from app.layers.visual_attributes import (XceptionBackbone, VisualAttributeHead, Denormalizer)
from app.utils.model_utils import ModelUtilities


class VisualAttrConfig:
    """
    Configuration for a single visual attribute prediction head in the model.

    This class encapsulates the dropout rate and L2 regularization factor
    for a specific visual attribute (e.g., 'L', 'a', 'contrast', etc.),
    allowing modular control over the behavior of each output head.
    """

    def __init__(self,
                 name: str,
                 dropout: float = 0.5,
                 reg: Optional[float] = 0.0001,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 ):
        """
        Initialize the configuration for a visual attribute.

        :param name: The name of the visual attribute (e.g., 'L', 'entropy').
        :param dropout: Dropout rate to apply before the Dense layer for this attribute.
                        Must be a float between 0.0 and 1.0. Default is 0.5.
        :param reg: L2 regularization factor to apply to the Dense layer's kernel.
                    If None, no regularization is applied. Default is 0.0001.
        """
        self.name = name
        self.dropout = dropout
        self.reg = reg
        self.mean = mean
        self.std = std

    def __repr__(self):
        return (f"VisualAttrConfig(name='{self.name}', dropout={self.dropout}, "
                f"reg={self.reg}, mean={self.mean}, std={self.std})")


@register_keras_serializable(package='VisualAttribute', name='VisualAttributePredictor')
class VisualAttributePredictor(Model, ModelUtilities):
    def __init__(self,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 initialize_with_imagenet: bool = False,
                 attribute_config: Optional[List[VisualAttrConfig]] = None,
                 input_name: str = 'image',
                 name='visual_attr_predictor', **kwargs):
        super().__init__(name=name,  **kwargs)

        self.input_name = input_name
        self.image_shape = image_shape
        self.base_model = XceptionBackbone(image_shape=image_shape,
                                           initialize_with_imagenet=initialize_with_imagenet)
        self.global_pool = GlobalAveragePooling2D(name='avg_pool')

        self.attribute_config = attribute_config or [
            VisualAttrConfig('L', 0.5, 0.01, mean=56.16887621, std=3.416602627),
            VisualAttrConfig('a', 0.5, 0.0001, mean=-13.3509608, std=4.958611063),
            VisualAttrConfig('b', 0.5, 0.0001, mean=42.68991623, std=4.907432068),
            VisualAttrConfig('contrast', 0.4, 0.0001, mean=130.8824943, std=49.53392233),
            VisualAttrConfig('correlation', 0.1, 0.0001, mean=0.983804451, std=0.006855397),
            VisualAttrConfig('energy', 0.5, 0.0001, mean=0.249356799, std=0.015899871),
            VisualAttrConfig('entropy', 0.5, 0.001, mean=6.006077151, std=0.155426234),
            VisualAttrConfig('homogeneity', 0.5, 0.0001, mean=0.547540533, std=0.066137278),
            VisualAttrConfig('uniformity', 0.5, 0.0001, mean=0.065048235, std=0.00932834),
            VisualAttrConfig('equivalent_diameter', 0.5, 0.001, mean=10.76372478, std=2.24162268),
            VisualAttrConfig('eccentricity', 0.5, 0.0001, mean=0.548399747, std=0.070028959),
            VisualAttrConfig('feret_diameter_max', 0.5, 0.001, mean=12.65424064, std=2.836734251),
            VisualAttrConfig('filled_area', 0.5, 0.001, mean=101.5903847, std=45.4936738),
            VisualAttrConfig('perimeter', 0.5, 0.01, mean=39.84774916, std=9.143521734),
            VisualAttrConfig('roundness', 0.5, 0.01, mean=0.729736846, std=0.0604815),
        ]

        self.initialize_with_imagenet = initialize_with_imagenet
        self.heads = self.build_attribute_heads()
        self.denorm_heads = self.build_denormalized_heads()

        self.built = False
        if image_shape is not None:
            self.build((None, *image_shape))
            self.built = True

        self.freeze_layers()

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, dict):
            inputs = inputs[self.input_name]

        x = self.base_model(inputs)
        x = self.global_pool(x)
        return {head.name: head(x) for head in self.heads}

    def build_attribute_heads(self):
        heads = []
        for attr_config in self.attribute_config:
            heads.append(VisualAttributeHead(visual_attribute=attr_config.name,
                                             dropout_rate=attr_config.dropout,
                                             reg_factor=attr_config.reg))
        return heads

    def build_denormalized_heads(self):
        heads = {}
        for attr_config in self.attribute_config:
            heads.update({
                attr_config.name: Denormalizer(mean=attr_config.mean,
                                                   std=attr_config.std,
                                                   name=f"{attr_config.name}")
            })
        return heads

    @property
    def trainable_layers(self):
        return [self.base_model, self.global_pool, *self.heads]

    def visual_attributes(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs[self.input_name]

        norm_preds = self.call(inputs)
        return {
            name: self.denorm_heads.get(f"{name}")(pred)
            for name, pred in norm_preds.items()
        }

    def set_weights_from_XceptionVisualNet(self, model):
        """
        Loads pretrained weights from another instance of `XceptionVisualNet` into this model.

        This method assumes:
        - The first 133 layers are part of the Xception backbone and are matched by their index.
        - The remaining layers (visual attribute prediction heads) are matched by name.

        :param model: An instance of `XceptionVisualNet` from which to copy the weights.
        :raises ValueError: If layer alignment or weight setting fails due to shape mismatch or missing layers.
        """

        local_xception_layers = list(self.get_inner_layers().values())[:133]
        local_attribute_layers = list(self.get_inner_layers().values())[133:]

        parent_xception_layers = model.layers[:133]
        name_of_parent_layers = [layer.name for layer in model.layers]

        for idx,layer in enumerate(local_xception_layers):
            layer.set_weights(weights=parent_xception_layers[idx].weights)

        for layer in local_attribute_layers:
            name = layer.name
            if name in name_of_parent_layers:
                layer.set_weights(weights=model.get_layer(name=name).get_weights())

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_shape': self.image_shape,
            'input_name': self.input_name,
            'initialize_with_imagenet': self.initialize_with_imagenet,
        })
        return config