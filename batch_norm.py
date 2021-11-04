import numpy as np


def batch_norm(
    input_layer: np.ndarray, betta: float = 0, gamma: float = 1
) -> np.ndarray:
    mean = input_layer.mean()
    std = input_layer.std()

    if std == 0:
        raise RuntimeError(
            "Standart deviation are zero (which leads to nan value) check your weight initialisation!"
        )

    return ((input_layer - mean) / std) * gamma + betta
