import numpy as np
from pytest import fixture

from activation_func import pixel_wise_softmax, relu, softmax


@fixture
def input_layer():
    return np.random.randint(low=-10, high=10, size=(10, 10, 5))


@fixture
def input_layer_three_classes():
    return np.array([1, 5, 2])


@fixture
def input_layer_pixel_wise():
    return np.array([[[1, 5, 2], [1, 5, 2], [1, 5, 2]]])


def test_relu(input_layer):
    output_layer = relu(input_layer)
    assert np.all(output_layer >= 0)


def test_softmax(input_layer_three_classes):
    output_layer = softmax(input_layer_three_classes)
    assert np.all(
        abs(output_layer - np.array([0.01714783, 0.93623955, 0.04661262])) < 1e-7
    )


def test_pixel_wise_softmax(input_layer_pixel_wise):
    output_layer = pixel_wise_softmax(input_layer_pixel_wise)
    assert np.all(
        abs(
            output_layer
            - np.array(
                [
                    [0.01714783, 0.93623955, 0.04661262],
                    [0.01714783, 0.93623955, 0.04661262],
                    [0.01714783, 0.93623955, 0.04661262],
                ]
            )
        )
        < 1e-6
    )
