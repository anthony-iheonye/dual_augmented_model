import tensorflow as tf
from tensorflow.keras.optimizers import (Adam, SGD, RMSprop)


class VisualAttrCompileConfig:
    """
    Configuration for compiling a VisualAttributePredictor Model.

    Encapsulates the optimizer, loss functions, metrics, and monitoring parameters
    for training models that predict visual attributes (e.g., color, texture, shape).

    :param learning_rate: Learning rate for the optimizer.
    :param momentum: Momentum factor (used in optimizers like SGD or RMSProp).
    :param optimizer: Optimizer name ('adam', 'sgd', or 'rmsprop').
    :param metric_name: Evaluation metric ('mae', 'mse', 'rmse', or 'mape').
    :param visual_attributes: Tuple of visual attribute names to be predicted.
    """
    def __init__(self,
                 learning_rate: float = 1e-5,
                 momentum: float = 0.99,
                 optimizer: str = 'Adam',
                 metric_name: str = 'mae',
                 visual_attributes: tuple = ('L',
                                             'a',
                                             'b',
                                             'contrast',
                                             'correlation',
                                             'energy',
                                             'entropy',
                                             'homogeneity',
                                             'uniformity',
                                             'equivalent_diameter', 'eccentricity',
                                             'feret_diameter_max',
                                             'filled_area',
                                             'perimeter',
                                             'roundness'),
                 ):
        """

        :param learning_rate: Learning for  the optimizer
        :param momentum: Momentum factor (used in optimizers like SGD or RMSProp).
        :param optimizer: Optimizer name ('adam', 'sgd', or 'rmsprop').
        :param metric_name: (str) Name of the metric to be used. The metric name could be either be
            'mse' for mean squared error, 'mae' for mean absolute error, 'rmse' for root mean squared
            error or 'mape' for mean absolute percentage error.
        :param visual_attributes:(tuple) Tuple of visual attribute names to be predicted..
        """

        if not isinstance(optimizer, str) or optimizer.lower() not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError(f"Optimizer must be 'adam', 'sgd', or 'rmsprop'")

        if not isinstance(metric_name, str) or metric_name.lower() not in ['mae', 'mse', 'rmse', 'mape']:
            raise ValueError(f"Metric_name must be 'mae', 'mse', 'rmse', or 'mape'")

        self.visual_attributes = visual_attributes

        # Optimizer
        optimizer_name = optimizer.lower()

        optimizer = {'adam': Adam(learning_rate=learning_rate),
                     'sgd': SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True),
                     'rmsprop': RMSprop(learning_rate=learning_rate, rho=0.9, momentum=momentum)}
        self.optimizer = optimizer[optimizer_name]

        # Loss function
        mse = tf.keras.losses.MeanSquaredError(name='mse')
        mae = tf.keras.losses.MeanAbsoluteError(name='mae')

        self.loss = {'L': mse,
                     'a': mse,
                     'b': mse,
                     'contrast': mae,
                     'correlation': mae,
                     'energy': mse,
                     'entropy': mse,
                     'homogeneity': mse,
                     'uniformity': mse,
                     'equivalent_diameter': mse,
                     'eccentricity': mse,
                     'feret_diameter_max': mse,
                     'filled_area': mse,
                     'perimeter': mse,
                     'roundness': mse}

        for key in list(self.loss.keys()):
            if key not in visual_attributes:
                self.loss.pop(key, None)

        # Metrics
        self.metric_name = metric_name
        self.metrics = {}

        if self.metric_name.lower() == 'mae':
            self.metrics = {'L': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'a': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'b': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'contrast': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'correlation': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'energy': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'entropy': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'uniformity': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'homogeneity': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'eccentricity': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'roundness': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'equivalent_diameter': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'feret_diameter_max': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'filled_area': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            'perimeter': tf.keras.metrics.MeanAbsoluteError(name='mae')}

        elif self.metric_name.lower() == 'rmse':
            self.metrics = {'L': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'a': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'b': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'contrast': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'correlation': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'energy': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'entropy': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'uniformity': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'homogeneity': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'eccentricity': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'roundness': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'equivalent_diameter': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'feret_diameter_max': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'filled_area': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            'perimeter': tf.keras.metrics.RootMeanSquaredError(name='rmse')}

        elif self.metric_name == 'mse':
            self.metrics = {'L': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'a': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'b': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'contrast': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'correlation': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'energy': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'entropy': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'uniformity': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'homogeneity': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'eccentricity': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'roundness': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'equivalent_diameter': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'feret_diameter_max': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'filled_area': tf.keras.metrics.MeanSquaredError(name='mse'),
                            'perimeter': tf.keras.metrics.MeanSquaredError(name='mse')}

        elif self.metric_name == 'mape':
            self.metrics = {'L': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'a': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'b': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'contrast': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'correlation': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'energy': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'entropy': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'uniformity': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'homogeneity': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'eccentricity': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'roundness': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'equivalent_diameter': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'feret_diameter_max': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'filled_area': tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                            'perimeter': tf.keras.metrics.MeanAbsolutePercentageError(name='mape')}

        # eliminate metrics for properties that are not in the pea_data.visual_attributes tuple
        for key in list(self.metrics.keys()):
            if key not in visual_attributes:
                self.metrics.pop(key, None)

        self.parameters_to_monitor = {'loss': ('loss',),
                                      'filled_area': ('filled_area_loss', 'filled_area_' + metric_name),
                                      'equivalent_diameter': ('equivalent_diameter_loss',
                                                              'equivalent_diameter_' + metric_name),
                                      'feret_diameter_max': ('feret_diameter_max_loss', 'feret_diameter_max_' + metric_name),
                                      'perimeter': ('perimeter_loss', 'perimeter_' + metric_name),
                                      'roundness': ('roundness_loss', 'roundness_' + metric_name),
                                      'eccentricity': ('eccentricity_loss', 'eccentricity_' + metric_name),
                                      'L': ('L_loss', 'L_' + metric_name),
                                      'a': ('a_loss', 'a_' + metric_name),
                                      'b': ('b_loss', 'b_' + metric_name),
                                      'contrast': ('contrast_loss', 'contrast_' + metric_name),
                                      'correlation': ('correlation_loss', 'correlation_' + metric_name),
                                      'energy': ('energy_loss', 'energy_' + metric_name),
                                      'entropy': ('entropy_loss', 'entropy_' + metric_name),
                                      'uniformity': ('uniformity_loss', 'uniformity_' + metric_name),
                                      'homogeneity': ('homogeneity_loss', 'homogeneity_' + metric_name)}

        for key in list(self.parameters_to_monitor.keys()):
            if key not in visual_attributes + ('loss',):
                self.parameters_to_monitor.pop(key, None)