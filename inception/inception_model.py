from tensorflow.keras import Model, Input
from keras.src.engine.keras_tensor import KerasTensor
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, Flatten, Dense, Concatenate, Conv2DTranspose
from typing import Union


class Inception(Model):  # pylint: disable=too-few-public-methods
    """
    Class creates model from given parameters.
    """
    def __init__(self,
                 inputs: tuple | list | KerasTensor = (299, 299, 1),
                 params: dict | None = None,
                 dense_end: bool = False) -> None:
        """
        :param inputs: (tuple | list | keras_tensor, defaults to (299, 299, 1)) Shape of input or input tensor.
        :param params: (dict | None, defaults to None) Model parameters. If None default parameters are used.
        :param dense_end: (bool, defaults to False) If True, adds a Dense layer at the end of the model. Valid only when params == None.
        """
        if params is None:
            from inception.inception_params import model_params
            params = model_params
            if dense_end:
                params['model'].extend(params['dense_end'])
        self.params = params

        if not isinstance(inputs, KerasTensor):
            inputs = Input(inputs)
        outputs = self._parse_params(inputs, params['model'])
        super().__init__(inputs=inputs, outputs=outputs)

    def _parse_params(self,
                      inputs: KerasTensor,
                      params: tuple | list) -> KerasTensor:
        """
        Parse model parameters.
        :param in: (KerasTensor) Input tensor.
        :param params: (tuple | list) Model parameters.
        :return: (KerasTensor) Output tensor.
        """
        x = inputs
        for layer_type, *layer_params in params:
            match layer_type:
                case 'a':  # Activation
                    an = layer_params[0]
                    x = Activation(an)(x)
                case 'ap2':  # AveragePooling2D
                    k, s, p = layer_params
                    x = AvgPool2D(pool_size=k,
                                  strides=s,
                                  padding=p)(x)
                case 'cba':  # Conv2D, BatchNormalization, Activation
                    f, k, s, p, an = layer_params
                    x = Conv2D(filters=f,
                               kernel_size=k,
                               strides=s,
                               padding=p)(x)
                    x = BatchNormalization()(x)
                    x = Activation(an)(x)
                case 'ctba':  # Conv2DTranspose, BatchNormalization, Activation
                    f, k, s, p, an = layer_params
                    x = Conv2DTranspose(filters=f,
                                        kernel_size=k,
                                        strides=s,
                                        padding=p)(x)
                    x = BatchNormalization()(x)
                    x = Activation(an)(x)
                case 'd':  # Dense / Linear
                    units = layer_params[0]
                    x = Dense(units=units)(x)
                case 'f':  # Flatten
                    x = Flatten()(x)
                case 'gap2':  # GlobalAveragePooling2D
                    x = GlobalAveragePooling2D()(x)
                case 'mp2':  # MaxPool2D
                    k, s, p = layer_params
                    x = MaxPool2D(pool_size=k,
                                  strides=s,
                                  padding=p)(x)
                case _:  # Other layers specified in model_param
                    submodules_outputs = []
                    try:
                        for submodule_params in self.params[layer_type]:
                            submodule_output_tensor = self._parse_params(x, submodule_params)
                            submodules_outputs.append(submodule_output_tensor)
                    except KeyError:
                        raise AttributeError(f'Unknown layer type: {layer_type}')
                    x = Concatenate(axis=-1)(submodules_outputs)
        return x
