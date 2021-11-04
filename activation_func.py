import numpy as np


def relu(input_layer: np.ndarray) -> np.ndarray:
    return np.maximum(input_layer, 0)


def pixel_wise_softmax(input_layer: np.ndarray) -> np.ndarray:
    i_h, i_w, _ = input_layer.shape
    output_layer = np.zeros(shape=input_layer.shape)

    for h in range(i_h):
        for w in range(i_w):
            output_layer[h, w, :] = softmax(input_layer[h, w])

    return output_layer


def softmax(input_layer: np.ndarray) -> np.ndarray:
    return np.exp(input_layer) / sum(np.exp(input_layer))
