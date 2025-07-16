from typing import Optional

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dropout, Dense)
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='VisualAttribute', name='VisualAttributeHead')
class VisualAttributeHead(Layer):
    def __init__(self,
                 visual_attribute:str,
                 dropout_rate: float = 0.0,
                 reg_factor: Optional[float] = None,
                 **kwargs):

        super().__init__(name=visual_attribute, **kwargs)
        self.visual_attribute = visual_attribute
        self.dropout_rate = dropout_rate
        self.reg_factor = reg_factor

        regularizer = self.get_regularizer(reg_factor)

        self.dropout = Dropout(rate=dropout_rate, name=f'dropout_{visual_attribute}')
        self.dense = Dense(units=1, name=f'{visual_attribute}', kernel_regularizer=regularizer)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        x = self.dropout(inputs)
        x = self.dense(x)
        return x

    @staticmethod
    def get_regularizer(factor: float = None):
        """
        produces the L2 regularization layer set to the assign regularization value

        :param factor: The regularization value
        :return: None if factor is set to None, or a regularization layer, if factor is a float.
        """
        if factor is None:
            return None
        else:
            return regularizers.l2(l2=factor)

    @property
    def trainable_layers(self):
        return self.dropout, self.dense

    def get_config(self):
        config = super().get_config()
        config.update({
            'visual_attribute': self.visual_attribute,
            'dropout_rate': self.dropout_rate,
            'reg_factor': self.reg_factor,
        })
        return config