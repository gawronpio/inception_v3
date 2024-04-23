from tensorflow.keras import Model, Input
from keras.src.engine.keras_tensor import KerasTensor
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, Flatten, Dense, Concatenate, Conv2DTranspose


class Inception(Model):  # pylint: disable=too-few-public-methods
    """
    Class creates model from given parameters.
    """
    def __init__(self,
                 inputs: tuple | list | KerasTensor = (299, 299, 1),
                 params: dict | None = None,
                 dense_end: bool = False,
                 name: str = 'Inception_V3') -> None:
        """
        :param inputs: (tuple | list | keras_tensor, defaults to (299, 299, 1)) Shape of input or input tensor.
        :param params: (dict | None, defaults to None) Model parameters. If None default parameters are used.
        :param dense_end: (bool, defaults to False) If True, adds a Dense layer at the end of the model. Valid only when params == None.
        :param name: (str, defaults to 'Inception_V3') Name of the model.
        """
        if params is None:
            from inception.inception_params import model_params
            params = model_params
            if dense_end:
                params['model'].extend(params['dense_end'])
        self.params = params

        if not isinstance(inputs, KerasTensor):
            inputs = Input(inputs, name=f'{name}_input')
        outputs = self._parse_params(inputs, params['model'], name=name)
        super().__init__(inputs=inputs, outputs=outputs, name=name)

    def _parse_params(self,
                      inputs: KerasTensor,
                      params: tuple | list,
                      name: str) -> KerasTensor:
        """
        Parse model parameters.
        :param in: (KerasTensor) Input tensor.
        :param params: (tuple | list) Model parameters.
        :param name: (str) Name of the model part.
        :return: (KerasTensor) Output tensor.
        """
        x = inputs
        for i, (layer_type, *layer_params) in enumerate(params, start=1):
            name_i = f'{name}_{i}'
            match layer_type:
                case 'a':  # Activation
                    an = layer_params[0]
                    x = Activation(an,
                                   name=f'{name_i}A')(x)
                case 'ap2':  # AveragePooling2D
                    k, s, p = layer_params
                    x = AvgPool2D(pool_size=k,
                                  strides=s,
                                  padding=p,
                                  name=f'{name_i}AP')(x)
                case 'cba':  # Conv2D, BatchNormalization, Activation
                    f, k, s, p, an = layer_params
                    x = Conv2D(filters=f,
                               kernel_size=k,
                               strides=s,
                               padding=p,
                               name=f'{name_i}C')(x)
                    x = BatchNormalization(name=f'{name_i}B')(x)
                    x = Activation(an,
                                   name=f'{name_i}A')(x)
                case 'ctba':  # Conv2DTranspose, BatchNormalization, Activation
                    f, k, s, p, an = layer_params
                    x = Conv2DTranspose(filters=f,
                                        kernel_size=k,
                                        strides=s,
                                        padding=p,
                                        name=f'{name_i}CT')(x)
                    x = BatchNormalization(name=f'{name_i}B')(x)
                    x = Activation(an,
                                   name=f'{name_i}A')(x)
                case 'd':  # Dense / Linear
                    units = layer_params[0]
                    x = Dense(units=units,
                              name=f'{name_i}D')(x)
                case 'f':  # Flatten
                    x = Flatten(name=f'{name_i}F')(x)
                case 'gap2':  # GlobalAveragePooling2D
                    x = GlobalAveragePooling2D(name=f'{name_i}GAP')(x)
                case 'mp2':  # MaxPool2D
                    k, s, p = layer_params
                    x = MaxPool2D(pool_size=k,
                                  strides=s,
                                  padding=p,
                                  name=f'{name_i}MP')(x)
                case _:  # Other layers specified in model_param
                    submodules_outputs = []
                    try:
                        for k, submodule_params in enumerate(self.params[layer_type], start=1):
                            submodule_output_tensor = self._parse_params(inputs=x,
                                                                         params=submodule_params,
                                                                         name=f'{name_i}_{k}{layer_type}')
                            submodules_outputs.append(submodule_output_tensor)
                    except KeyError:
                        raise AttributeError(f'Unknown layer type: {layer_type}')
                    x = Concatenate(axis=-1,
                                    name=f'{name_i}Concat-{layer_type}')(submodules_outputs)
        return x
