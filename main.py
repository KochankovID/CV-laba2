import numpy as np
from cv2 import cv2

from activation_func import pixel_wise_softmax, relu
from batch_norm import batch_norm
from convolution import fast_conv
from pooling import max_pooling

IMAGE = "img/test2.jpg"


if __name__ == "__main__":
    image = cv2.imread(IMAGE)

    filters = np.random.normal(size=(3, 3, 3, 5))

    output_layer_1 = fast_conv(image, filters)
    print("conv output shape:", output_layer_1.shape)

    output_layer_2 = batch_norm(output_layer_1)
    print("batch normalization output shape:", output_layer_2.shape)

    output_layer_3 = relu(output_layer_2)
    print("relu output shape:", output_layer_3.shape)

    output_layer_4 = max_pooling(output_layer_3, height=2, width=2)
    print("max pooling output shape:", output_layer_4.shape)

    output_layer_5 = pixel_wise_softmax(output_layer_4)
    print("pixel-wise soft max output shape:", output_layer_5.shape)

    cv2.imshow('image', image)
    cv2.waitKey()
