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
    def __init__(self, H, W, C, N, n_bottleneck, n_hidden_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bottleneck = n_bottleneck
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.n_hidden_units = n_hidden_units

    def build(self, input_shape):
        # TODO: correct way to do made for every pixel? output H * W * C?
        """
        Model is
        Image -> PixelCNN (bs, H, W, C * N) -> Flatten -> Dense layer (bottleneck reduce dimensionality)
        -> MADE (D = H x W x C variables for each channel each pixel)
        """
        self.layers_list = [PixelCNNModel(self.H, self.W, self.C, self.N, flat=True),
                            Flatten(),
                            Dense(self.n_bottleneck),
                            MADEModel(self.H * self.W * self.C, self.N, self.n_hidden_units)]
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        output shape is (bs, H * W * C, N)
        """
        for layer in self.layers_list:
            x = layer(x)
        x = tf.reshape(x, (-1, self.H, self.W, self.C, self.N))
        return x

# overwrite MADE
class PixelCNNMADE(MADE):
    def __init__(self, H=28, W=28, C=3, N=4, learning_rate=10e-4, n_bottleneck=512, n_hidden_units=124):
        """
        H, W, C image shape: height, width, channels
        N is number of values per variable
        """
        name = "PixelCNN-MADE"
        self.H = H
        self.W = W
        self.C = C
        self.n_bottleneck = n_bottleneck
        self.n_hidden_units = n_hidden_units
        # calls setup model and init optimiser
        super().__init__(name, N, self.H * self.W * self.C, n_hidden_units=124, one_hot=False, learning_rate=learning_rate, )

    def setup_model(self):
        """
        overwrite to pixelcnn-made model
        """
        # TODO: make sure pixel cnn uses factorised? but full joint should be ok too?
        # TODO: larger n hidden units in MADE?
        self.model = PixelCNNMADEModel(self.H, self.W, self.C, self.N, self.n_bottleneck, self.n_hidden_units)

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
        images[:, 0, 0, 0] = np.random.choice(self.N, size=n)
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C):
                    model_preds = self.forward_softmax(images)
                    # categorical over pixel values
                    pixel_dist = tfp.distributions.Categorical(probs=model_preds[:, h, w, c])
                    images[:, h, w, c] = pixel_dist.sample(1)
        return images
