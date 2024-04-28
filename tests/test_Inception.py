from inception.inception_model import Inception
import pytest


class TestInception:

    def test_default_parameters_and_input_shape(self, default_model):
        """
        Test if Inception model can be created with default parameters and input shape.
        :return: None.
        """
        # Given

        # When

        # Then
        assert isinstance(default_model, Inception)

    def test_custom_parameters_and_input_shape(self):
        """
        Test if Inception model can be created with custom parameters and input shape.
        :return: None.
        """
        # Given
        params = {
            'model': [
                ('ap2', 3, 2, 'same'),
                ('cba', 64, 3, 1, 'same', 'ReLU'),
                ('mp2', 2, 2, 'valid')
            ]
        }
        inputs = (224, 224, 3)

        # When
        model = Inception(inputs=inputs, params=params)

        # Then
        assert isinstance(model, Inception)

    def test_unknown_layer_type(self):
        """
        Test if Inception model raises an AttributeError for an unknown layer type.
        :return: None.
        """
        # Given
        params = {
            'model': [
                ('ap2', 3, 2, 'same'),
                ('cba', 64, 3, 1, 'same', 'ReLU'),
                ('unknown', 2, 2, 'valid')
            ]
        }

        # When / Then
        with pytest.raises(AttributeError):
            Inception(params=params)

    def test_default_model_layer_count(self, default_model):
        """
        Test if Inception model has the correct number of layers.
        :return: None.
        """
        # Given
        expected_layers = 347

        # When
        actual_layers = len(default_model.layers)

        # Then
        assert actual_layers == expected_layers

    def test_default_model_inputs_count(self, default_model):
        """
        Test if Inception model has the correct number of inputs.
        :return: None.
        """
        # Given
        expected_inputs = 1

        # When
        actual_inputs = len(default_model.inputs)

        # Then
        assert actual_inputs == expected_inputs

    def test_default_model_outputs_count(self, default_model):
        """
        Test if Inception model has the correct number of outputs.
        :return: None.
        """
        # Given
        expected_outputs = 1

        # When
        actual_outputs = len(default_model.outputs)

        # Then
        assert actual_outputs == expected_outputs
