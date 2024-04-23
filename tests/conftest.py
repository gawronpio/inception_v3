import pytest


@pytest.fixture(scope='module', autouse=True)
def turn_off_gpu():
    """
    Turn off GPU for all tests.
    :return: None
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@pytest.fixture(scope='module', autouse=False)
def default_model():
    """
    Prepare model for testing.
    :return: None
    """
    from inception.inception_model import Inception
    model = Inception()
    return model
