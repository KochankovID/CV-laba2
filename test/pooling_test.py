import numpy as np
from pytest import fixture

from pooling import max_pooling


@fixture
def input_layer():
    return np.random.randint(low=-10, high=10, size=(10, 10, 5))


def test_pooling(input_layer):
    output_layer = max_pooling(input_layer, height=2, width=2)
    assert len(output_layer.shape)
