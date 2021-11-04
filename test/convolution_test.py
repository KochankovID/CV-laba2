import numpy as np
from pytest import fixture

from convolution import conv, fast_conv


@fixture
def image_zeros():
    return np.zeros(shape=(10, 10, 3))


@fixture
def filter_zeros():
    return np.zeros(shape=(10, 10, 3, 5))


@fixture
def image_ones():
    return np.ones(shape=(10, 10, 3))


@fixture
def filters_ones():
    return np.ones(shape=(10, 10, 3, 5))


@fixture
def image():
    return np.random.normal(size=(10, 10, 3))


@fixture
def filters():
    return np.random.normal(size=(10, 10, 3, 5))


def test_zero_tensors(image_zeros, filter_zeros):
    output_layer = conv(image_zeros, filter_zeros)
    assert np.all(output_layer == np.zeros(shape=(1, 1, 5)))


def test_ones_tensors(image_ones, filters_ones):
    output_layer = conv(image_ones, filters_ones)
    assert np.all(output_layer == np.full(shape=(1, 1, 5), fill_value=10 * 10 * 3))


def test_equivalent(image, filters):
    output_layer_slow = conv(image, filters)
    output_layer_fast = fast_conv(image, filters)
    assert np.all(abs(output_layer_slow - output_layer_fast) < 0.1e-10)
