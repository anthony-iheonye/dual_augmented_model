from typing import Union, Optional, Dict, List, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import Layer


class XLayer(Layer):
    def __init__(self, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.trainable_layers = None


class ModelUtilities:
    def __init__(self):
        self.name: Optional[str] = None
        self.trainable_layers: Optional[list[XLayer]] = None
        self.built = False

    def get_inner_layers(self) -> Dict[str, XLayer]:
        """
        Recursively retrieve and return all basic (non-nested) layers in the model.
        Example of basic layers includes Dense, Conv2D, Dropout, MaxPool, Concat, etc.
        """
        layers = {}
        if not self.trainable_layers:
            return layers

        for layer in self.trainable_layers:
            self._collect_inner_layers(layer, layers)

        return layers

    def _collect_inner_layers(self, layer: XLayer, layers: Dict[str, XLayer]) -> None:
        """
        Recursively collects the most basic trainable layers from a nested layer structure.
        """

        # Base case: If the layer does not contain nested layers.
        if not hasattr(layer, 'trainable_layers') or not layer.trainable_layers:
            layers[layer.name] = layer
            return

        # Recursive case: dig into nested layers
        for sublayer in layer.trainable_layers:
            self._collect_inner_layers(sublayer, layers)

    def view_trainable_status(self, line_length=50):
        """
        View which layers in the model have frozen (trainable) or unfrozen
        (un-trainable) weights.

        :return: A printout containing the index (position of the layer
            in the model), the name, and trainable status of all the layers in the model
        """
        length = 50
        name_length = len(self.name)
        extra_ticks = line_length - length
        num_ticks = (length + extra_ticks - (name_length + 13)) // 2

        print(f"\n\n{'-' * num_ticks + self.name.upper() + ' LAYER STATUS' + '-' * num_ticks :^{length}}\n"
              f"{'INDEX': <7s}{'LAYER_NAME':<{28 + extra_ticks}s}{'TRAINABLE LAYER'}")

        for idx, (name, layer) in enumerate(self.get_inner_layers().items()):
            print(f"{str(idx ) +'.':<7}{name:<{28 + extra_ticks}s}{layer.trainable}")

    def freeze_layers(self, layers: Union[str, list, tuple] = None):
        """
        Freeze the weight of specific layers in the model. layers is optionally set to `None`
        :param layers: [str, list, tuple], Names of layers to freeze. A list or tuple containing the names of
            the layers whose weights are to be frozen. To freeze all the layers, set 'layers' to 'all'.
        """

        if layers is not None and not isinstance(layers, (str, list, tuple)):
            raise ValueError \
                (f"layers must be either be set to 'None' or a [str, list, tuple], got {type(layers).__name__}.")

        # Produce a Dict mapping layer name to layer object.
        model_layers = self.get_inner_layers()

        if layers is None:
            for layer in model_layers.values():
                layer.trainable = True
            return

            # Freeze all the weights in the model, if layers = 'all'
        if isinstance(layers, str):
            if layers.lower() == 'all':
                for layer in model_layers.values():
                    layer.trainable = False
                return

                # Unfreeze the weights of all layers
        for layer in model_layers.values():
            layer.trainable = True

        # Freeze the weights of specified layers
        for name in layers:
            layer = model_layers.get(name)
            if not layer:
                continue
            layer.trainable = False

        return

    def set_weights_from_layers(self, layers: Union[List[Layer], Tuple[Layer]]) -> None:
        """
        Load weights from a list of layers into the current model layers (by matching layer names).
        Args:
            layers (Union[List[Layer], Tuple[Layer]]): A collection of Keras Layer objects with weights to load.
        """

        if not self.built:
            raise ValueError(
                "You called `set_weights_from_layers` on a model that has not been built.\n"
                "Please run `model.build((None, height, width, channels))` before calling this method, "
                "or pass `input_shape` when instantiating the model to ensure it is built automatically."
            )

        inner_layers = self.get_inner_layers()

        for layer in layers:
            name = layer.name
            if name in inner_layers:
                inner_layers.get(name).set_weights(layer.get_weights())


class L2Regularizer:

    def __init__(self):
        """
        Returns an L2 regularization layer, set to the assign regularization value.
        """

    def __call__(self, factor: Optional[float]=None):
        """

        :param factor: The regularization value
        :return: None, if factor is set to None, or a regularization layer, if factor is a float.
        """
        if factor is None:
            return None
        else:
            return tf.keras.regularizers.l2(l2=factor)


class LayerExtractor:
    """
    Returns the lower_level layers in a model. It the input is a
    layer, this class returns the sub-layers, if any.
    """


    def __call__(self, model: Union[Model, XLayer]):
        """

        :param model: A tf.Keras Model or Layer object
        :return: A list containing the low-level layers.
        """

        layers = {}
        for layer in model.trainable_layers:
            layers.update(self._get_layers(layer, {}))
        return layers

    def _get_layers(self, layer, result):
        if not hasattr(layer, 'trainable_layers'):
            result[layer.name] = layer
            return result

        if len(layer.trainable_layers) == 0:
            return None

        for child_layer in layer.trainable_layers:
            self._get_layers(child_layer, result)

        return result


class UNetTFModule(tf.Module):
    """Create a tf.Module version of a UNet tf.keras model. The UNet module would have a segment method."""
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def segment(self, image: tf.Tensor):
        """Returns an RGB image containing only the objects of interest or foreground"""
        return self.model.segment(image)



