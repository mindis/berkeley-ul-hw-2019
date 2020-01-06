import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Dense, Flatten

from MADE import MADEModel, MADE
from pixelCNN import PixelCNNModel
from utils import tf_log_to_base_n, tf_log2, gather_nd


# wrap model so it can be called in MADE
class PixelCNNMADEModel(Model):
    def __init__(self, H, W, N, D, n_bottleneck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bottleneck = n_bottleneck
        self.H = H
        self.W = W
        self.N = N
        self.D = D

    def build(self, input_shape):
        # TODO: how to do made for every pixel? output H * W * C?
        self.layers_list = [PixelCNNModel(), Flatten(), Dense(self.n_bottleneck), MADEModel(self.H * self.W * self.N, self.D)]
        super().build(input_shape)

    def call(self, x, **kwargs):
        for layer in self.layers_list:
            x = layer(x)
        # outputs is (bs, H * W * N, D), reshape to image
        x_reshape = tf.reshape(x, (-1, self.H, self.W, self.N, self.D))
        return x_reshape

# overwrite MADE
class PixelCNNMADE(MADE):
    def __init__(self, H=28, W=28, C=3, D=3, N=4, learning_rate=10e-4, n_bottleneck=512):
        """
        H, W, C image shape: height, width, channels
        D is number of variables (RGB = 3)
        N is number of values per variable
        """
        name = "PixelCNN-MADE"
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.H = H
        self.W = W
        self.C = C
        self.n_bottleneck = n_bottleneck
        # calls setup model and init optimiser
        super().__init__(name, N, D, one_hot=False)

    def setup_model(self):
        # TODO: make sure pixel cnn uses factorised? but full joint should be ok too?
        # TODO: larger n hidden units in MADE?
        self.model = PixelCNNMADEModel(self.H, self.W, self.N, self.D, self.n_bottleneck)

    # copied from pixelCNN

    def eval_batch(self, X, bs=128):
        """
        computes forward pass then logprob on the outputs
        X is batched in to handle large data
        note this returns mean logprob over batch
        """
        # for small batch just eval it
        if len(X) <= bs:
            return self.eval(X)
        # otherwise evaluate in batches
        neg_logprobs_bits = []
        # eval in batches
        for i in range(len(X) // bs):
            neg_logprobs_bits.append(self.eval(X[i * bs: (i + 1) * bs]))
        # mean of batches
        mean_nll = tf.reduce_mean(neg_logprobs_bits)
        # deal with leftover data if not a multiple of batch size
        extra_data = X[(len(X) // bs) * bs:]
        if len(extra_data) == 1:
            # add batch dimension if single data
            extra_data = [extra_data]
        if len(extra_data) > 0:
            # evaluate extra data
            extra_data_nll_bits = self.eval(extra_data)
            # weight the mean of the batches and extra data
            n_extra = len(extra_data)
            mean_nll = ((len(X) - n_extra) / len(X)) * mean_nll + (n_extra / len(X)) * extra_data_nll_bits
        return mean_nll

    def get_samples(self, n):
        """
        Generation is done from blank image (all 0s), we then sample R channel
        of first pixel, then G then B and then for second pixel etc.
        We batch this for efficiency.
        n is number of samples to draw.
        """
        images = np.zeros((n, self.H, self.W, self.C))
        # start with random values for first channel of first pixel (this is updated in first pass)
        images[:, 0, 0, 0] = np.random.choice(self.D, size=n)
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C):
                    model_preds = self.forward_softmax(images)
                    # categorical over pixel values
                    pixel_dist = tfp.distributions.Categorical(probs=model_preds)
                    images[:, h, w, c] = pixel_dist.sample(1)
        return images
