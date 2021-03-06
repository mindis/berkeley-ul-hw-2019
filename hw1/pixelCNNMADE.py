import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Lambda

from MADE import MADEModel, MADE, one_hot_inputs
from pixelCNN import PixelCNNModel
from utils import tf_log_to_base_n, tf_log2, gather_nd

# wrap model so it can be called in MADE
class PixelCNNMADEModel(Model):
    def __init__(self, H, W, C, N, D, n_hidden_units, factorised, *args, **kwargs):
        """
        H, W, C image shape: height, width, channels
        N is number of values per variable
        D is number of variables
        """
        super().__init__(*args, **kwargs)
        self.H = H
        self.W = W
        self.C = C
        self.D = D
        self.N = N
        self.n_hidden_units = n_hidden_units
        self.factorised = factorised

    def build(self, input_shape):
        """
        Model is
        Image -> PixelCNN (bs, H, W, C * N) -> Flatten -> aux
        (Image one hot, aux) -> MADE (D variables)
        """
        self.pixelcnn_model = PixelCNNModel(self.H, self.W, self.C, self.N, factorised=self.factorised)#, flat=True)
        # we have auxiliary variables of the flattened image shape
        self.made_model = MADEModel(self.D, self.N, self.n_hidden_units, N_aux=self.D*self.N)

    def call(self, inputs, **kwargs):
        """
        output shape is (bs, D, N)
        """
        # get pixelCNN outputs
        x = self.pixelcnn_model(tf.cast(inputs, tf.float32) / self.N)
        x = tf.nn.relu(x)
        # we input the pixelCNN outputs as auxiliary variables to MADE
        # reshape such that each pixel is a data point in a batch of size (n_images_in_batch * image_size_flat)
        # each pixel is then passed through MADE
        aux_input = tf.reshape(x, (-1, self.D * self.N))
        # we input the current pixel's channels one-hot to MADE
        x_one_hot = one_hot_inputs(inputs, self.D, self.N)
        # concat inputs and aux inputs for MADE
        x = self.made_model((x_one_hot, aux_input))
        x = tf.reshape(x, (-1, self.H, self.W, self.C, self.N))
        return x

# overwrite MADE
class PixelCNNMADE(MADE):
    def __init__(self, H=28, W=28, C=3, N=4, D=3, learning_rate=10e-4, n_hidden_units=256):
        """
        H, W, C image shape: height, width, channels
        N is number of values per variable
        D is number of variables, here it's one for each channel for each pixel
        """
        name = "PixelCNN-MADE"
        self.H = H
        self.W = W
        self.C = C
        self.n_hidden_units = n_hidden_units
        # calls setup model and init optimiser
        # D (# vars) is H x W x C
        # We don't want input to be one_hot as it is passed to pixelCNN, we then one_hot pixelcnn output before MADE
        super().__init__(name, N, D, n_hidden_units=124, one_hot=False,
                         learning_rate=learning_rate, grad_clip=0.1)

    def setup_model(self):
        """
        overwrite to pixelcnn-made model
        """
        self.model = PixelCNNMADEModel(self.H, self.W, self.C, self.N, self.D,
                                       self.n_hidden_units, factorised=True)

    def eval_dataset(self, X, bs=128):
        """
        :param X: a tf.data.Dataset
        computes eval on a tf dataset
        returns float of mean loss on dataset
        """
        n_data = 0
        weighted_sum = 0
        for batch in X.shuffle(bs * 2).batch(bs):
            n = len(batch)
            loss = self.eval(batch).numpy()
            weighted_sum += loss * n
            n_data += n
        return weighted_sum / n_data

    def get_samples(self, n, seed=123):
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
                    images[:, h, w, c] = pixel_dist.sample(1, seed=seed)
        return images
