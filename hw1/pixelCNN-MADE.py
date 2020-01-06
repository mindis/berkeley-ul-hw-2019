import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow_core.python.keras.layers import Dense, Flatten

from MADE import MADEModel
from pixelCNN import PixelCNNModel
from utils import tf_log_to_base_n

# eval, eval_batch, train_step, forward, loss, sample
class PixelCNNMADE:
    def __init__(self, H=28, W=28, C=3, n_vals=4, learning_rate=10e-4, n_bottleneck=256):
        """
        H, W, C image shape: height, width, channels
        n_vals the number of values each channel can take on
        """
        self.name = "PixelCNN"
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.H = H
        self.W = W
        self.C = C
        self.n_vals = n_vals
        self.n_bottleneck = n_bottleneck
        self.setup_model()

    def setup_model(self):
        # TODO: test rewritten MADE: now accepts different input size to output
        # TODO: make sure pixel cnn uses factorised?
        # TODO: larger n hidden units in MADE?
        pixelcnn = PixelCNNModel()
        flatten = Flatten()(pixelcnn)
        dense = Dense(self.n_bottleneck)(flatten)
        made = MADEModel(3, 4)
        self.model = made(dense)


