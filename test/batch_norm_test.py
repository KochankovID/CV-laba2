import numpy as np
from pytest import fixture

from batch_norm import batch_norm


@fixture
def input_layer():
    return np.random.randint(low=-10, high=10, size=(10, 10, 5))


def test_batch_norm(input_layer):
    output_layer = batch_norm(input_layer)
    assert (output_layer.mean() < 1e-11) and (abs(output_layer.std() - 1) < 1e-11)
