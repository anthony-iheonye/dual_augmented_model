import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='VisualAttribute', name='Denormalizer')
class Denormalizer(Layer):
    def __init__(self, mean: float = 1.0, std: float = 0.0, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        return inputs * self.std + self.mean

    def get_config(self):
        config = super().get_config()
        config.update({
            'mean': self.mean,
            'std': self.std,
            'name': self.name
        })