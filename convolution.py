import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm


def conv(input_layer: np.ndarray, conv_filters: np.ndarray, step=1) -> np.ndarray:
    _conv_check(input_layer, conv_filters, step)

    f_h, f_w, f_c, f_num = conv_filters.shape
    i_h, i_w, i_c = input_layer.shape

    o_h = int((i_h - f_h) / step + 1)
    o_w = int((i_w - f_w) / step + 1)

    output_layer = np.zeros(shape=(o_h, o_w, f_num))

    for i in tqdm(range(f_num), desc="conv"):
        for h in range(o_h):
            for w in range(o_w):

                sum = 0

                for y in range(f_h):
                    for x in range(f_w):
                        for c in range(f_c):
                            sum += (
                                input_layer[h * step + y][w * step + x][c]
                                * conv_filters[y][x][c][i]
                            )

                output_layer[h][w][i] = sum

    return output_layer


def fast_conv(input_layer: np.ndarray, conv_filters: np.ndarray) -> np.ndarray:
    _conv_check(input_layer, conv_filters)
    f_h, f_w, f_c, f_num = conv_filters.shape
    i_h, i_w, i_c = input_layer.shape

    o_h = int((i_h - f_h) + 1)
    o_w = int((i_w - f_w) + 1)

    strided = as_strided(
        input_layer,
        (o_h, o_w, f_h, f_w, i_c),
        input_layer.strides[:2] + input_layer.strides,
    )
    return np.tensordot(strided, conv_filters, axes=3)


def _conv_check(input_layer: np.ndarray, conv_filters: np.ndarray, step=1):
    if step < 1:
        raise RuntimeError("Step can't be less than 1")

    if len(input_layer.shape) != len(conv_filters.shape[:-1]):
        raise RuntimeError("Image shape doesn't match filter shape")

    if any(
        input_layer.shape[i] < conv_filters.shape[i]
        for i in range(len(input_layer.shape))
    ):
        raise RuntimeError(
            f"Input image less than filter {input_layer.shape} {conv_filters.shape[:-1]}"
        )
