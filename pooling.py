import numpy as np


def max_pooling(input_layer: np.ndarray, height: int, width: int) -> np.ndarray:
    if not 0 < height <= input_layer.shape[0] or not 0 < width <= input_layer.shape[1]:
        raise RuntimeError("Wrong kernel size!")

    output_layer = np.empty(
        shape=(
            int(input_layer.shape[0] / height),
            int(input_layer.shape[1] / width),
            input_layer.shape[2],
        )
    )

    for h in range(output_layer.shape[0]):
        for w in range(output_layer.shape[1]):
            output_layer[h, w] = np.max(
                input_layer[h * height : (h + 1) * height, w * width : (w + 1) * width],
                axis=(0, 1),
            )

    return output_layer
