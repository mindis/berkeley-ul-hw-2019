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
    def __init__(self, H, W, C, N, n_hidden_units, factorised, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.n_hidden_units = n_hidden_units
        self.factorised = factorised

    def build(self, input_shape):
        """
        Model is
        Image -> PixelCNN (bs, H, W, C * N) -> Flatten -> Dense layer (bottleneck reduce dimensionality)
        -> MADE (D = H x W x C variables for each channel each pixel)
        """
        self.layers_list = [PixelCNNModel(self.H, self.W, self.C, self.N, factorised=self.factorised,
                                          flat=True),
                            Flatten(),
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
    def __init__(self, H=28, W=28, C=3, N=4, learning_rate=10e-4, n_hidden_units=124):
        """
        H, W, C image shape: height, width, channels
        N is number of values per variable
        """
        name = "PixelCNN-MADE"
        self.H = H
        self.W = W
        self.C = C
        self.n_hidden_units = n_hidden_units
        # calls setup model and init optimiser
        # D (# vars) is H x W x C
        super().__init__(name, N, self.H * self.W * self.C, n_hidden_units=124, one_hot=False,
                         learning_rate=learning_rate)

    def setup_model(self):
        """
        overwrite to pixelcnn-made model
        """
        # we don't want factorised bc MADE part of this model is to capture dependencies between channels
        self.model = PixelCNNMADEModel(self.H, self.W, self.C, self.N,
                                       self.n_hidden_units, factorised=False)

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


def ds_get_masks(nrof_units, nrof_layers, nrof_dims, nrof_aux, nrof_bins):
    m = []
    m0 = np.repeat(np.arange(nrof_dims), nrof_bins)
    m += [m0]
    for i in range(nrof_layers):
        rep = int(np.ceil(nrof_units / ((nrof_dims - 1))))
        mx = np.repeat(np.arange(nrof_dims - 1), rep)[:nrof_units]
        m += [mx]

    mask = []
    for i in range(len(m) - 1):
        msk = m[i + 1][:, None] >= m[i][None, :]
        cx = np.ones((msk.shape[0], nrof_aux))
        msk2 = np.concatenate((cx, msk), axis=1)
        mask += [msk2.T]
    msk = m0[:, None] > m[-1][None, :]
    cx = np.ones((msk.shape[0], nrof_aux))
    msk2 = np.concatenate((cx, msk), axis=1)
    mask += [msk2.T]

    return mask


if __name__ == "__main__":
    nrof_units, nrof_layers, nrof_dims, nrof_bins = 6, 2, 3, 4
    ds_masks = ds_get_masks(nrof_units, nrof_layers, nrof_dims, 0, nrof_bins)
    print(ds_masks[2])

    from MADE import get_mask_made, ordered_unit_number, sample_unit_numbers
    order_units = ordered_unit_number(nrof_dims, nrof_bins)
    sample_units = sample_unit_numbers(nrof_units, 1, nrof_dims)
    layer_units = np.repeat(np.arange(1, nrof_dims+1), nrof_units // nrof_dims+1)[:nrof_units]
    print(sample_units)
    mask = get_mask_made(sample_units, order_units)
    print(mask)