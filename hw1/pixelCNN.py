import numpy as np
import tensorflow as tf

from utils import tf_log2, gather_nd


# TODO: copy mostly from tf.keras.Conv2D
class MaskedCNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


"""
Type A takes context and previous channels (but not its own channel)
"""
class PixelCNNTypeA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


"""
Type B takes context, prev channels and connected to own channel.
"""
class PixelCNNTypeB(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


class PixelCNN:
    def __init__(self):
        pass

    def setup_model(self):
        pass


if __name__ == "__main__":
    pass