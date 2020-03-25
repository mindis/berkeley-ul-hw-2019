import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from pixelCNN import PixelCNNModel
from MADE import sample_unit_numbers, input_unit_numbers, ordered_unit_number, get_mask_made, MADELayer

#
class DenseMasked(tf.keras.layers.Layer):
    def __init__(self, nrof_units, mask, activation=None, **kwargs):
        super(DenseMasked, self).__init__(**kwargs)
        self._mask = mask
        self._activation = activation
        self.nrof_units = nrof_units

    def build(self, input_shape):
        nrof_inputs = input_shape[1]
        self._kernel = tf.Variable(tf.initializers.glorot_normal()((nrof_inputs, self.nrof_units)), dtype=tf.float32, trainable=True)
        self._bias = tf.Variable(tf.initializers.zeros()((self.nrof_units,)), dtype=tf.float32, trainable=True)

    def call(self, inputs, **kwargs):
        y = tf.tensordot(inputs, self._kernel * self._mask, 1) + self._bias
        y = self._activation(y) if self._activation else y
        return y


def get_masks(nrof_units, nrof_layers, nrof_dims, nrof_aux, nrof_bins, aux_all):
    m = []
    m0 = np.repeat(np.arange(nrof_dims), nrof_bins)
    m += [m0]
    for i in range(nrof_layers):
        rep = int(np.ceil(nrof_units / ((nrof_dims - 1))))
        mx = np.repeat(np.arange(nrof_dims - 1), rep)[:nrof_units]
        m += [mx]

    mask = []
    for i in range(len(m) - 1):
        if i == 0 or aux_all:
            msk = m[i + 1][:, None] >= m[i][None, :]
            cx = np.ones((msk.shape[0], nrof_aux))
            msk2 = np.concatenate((cx, msk), axis=1)
        else:
            msk2 = np.array(m[i + 1][:, None] >= m[i][None, :], dtype=float)
        mask += [msk2.T]
    msk2 = np.array(m0[:, None] > m[-1][None, :], dtype=float)
    if aux_all:
        cx = np.ones((msk2.shape[0], nrof_aux))
        msk2 = np.concatenate((cx, msk2), axis=1)
    mask += [msk2.T]

    return mask


class DS_PixelCNN_MADE_Model(tf.keras.Model):
    def __init__(self, H, W, C, N, D, n_hidden_units=132, n_layers=2):
        super(DS_PixelCNN_MADE_Model, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.D = D
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.aux_all = False

    def build(self, input_shape):
        self.pixelCNN = PixelCNNModel(self.H, self.W, self.C, self.N, True)
        # made
        masks = get_masks(self.n_hidden_units, self.n_layers, self.D, self.N * self.D, self.N, self.aux_all)
        hidden = [self.n_hidden_units] * self.n_layers + [self.D * self.N]
        self.made_layers = []
        prev_unit_numbers = input_unit_numbers(self.D, self.N, self.N * self.D)
        for i, h in enumerate(hidden):
            activation = tf.nn.relu if i < self.n_layers else None
            # self.made_layers.append(DenseMasked(h, masks[i], activation=activation))
            self.made_layers.append(MADELayer(h, prev_unit_numbers, self.D, activation=activation, is_output=i<self.n_layers))
            prev_unit_numbers = self.made_layers[-1].unit_numbers

    def call(self, inputs, training=None, mask=None):
        x_made = tf.reshape(tf.one_hot(tf.cast(inputs, tf.int32), depth=4), (-1, self.N * self.D))
        x_pixelcnn = tf.reshape(inputs, (-1, self.H, self.W, self.C))
        aux = self.pixelCNN(x_pixelcnn)
        aux_rshp = tf.reshape(aux, (-1, self.N * self.D))
        x = x_made
        if not self.aux_all:
            x = tf.concat([aux_rshp, x], -1)
        for layer in self.made_layers:
            if self.aux_all:
                x = tf.concat([aux_rshp, x], -1)
            x = layer(x)
        y_made_rshp = tf.reshape(x, (-1, self.H, self.W, self.C, self.N))
        return y_made_rshp


class DS_PixelCNN_MADE:
    def __init__(self, H=28, W=28, C=3, N=4, D=3, learning_rate=10e-4, n_hidden_units=124):
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.D = D
        self.name = "DS_PixelCNN_MADE"
        self.n_hidden_units = n_hidden_units
        self.model = DS_PixelCNN_MADE_Model(H, W, C, N, D, n_hidden_units=n_hidden_units)
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def eval(self, x):
        logits = self.model(x)
        return self.loss(x, logits)

    def loss(self, x, logits):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(x, tf.int32), logits=logits)
        loss = tf.reduce_mean(losses) * np.log2(np.e)
        return loss

    def train_step(self, X_train):
        with tf.GradientTape() as tape:
            logprob = self.eval(X_train)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob.numpy()

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

    def forward_softmax(self, x):
        return tf.nn.softmax(self.model(x))

#
# if __name__ == "__main__":
#     nrof_dims, nrof_bins = 3, 4
#     nrof_units, nrof_layers = 12, 2
#     n_aux = 5
#     ds_masks = get_masks(nrof_units, nrof_layers, nrof_dims, n_aux, nrof_bins)
#
#     in_units = input_unit_numbers(nrof_dims, nrof_bins, n_aux)
#     out_units = ordered_unit_number(nrof_dims, nrof_bins) - 1
#     sample_units = sample_unit_numbers(nrof_units, 1, nrof_dims, ordered=True)
#     sample_units2 = sample_unit_numbers(nrof_units, 1, nrof_dims, ordered=True)
#     my_masks = []
#     my_masks.append(get_mask_made(in_units, sample_units, False))
#     my_masks.append(get_mask_made(sample_units, sample_units2, False))
#     my_masks.append(get_mask_made(sample_units2, out_units, False))
#     for m, d in zip(my_masks, ds_masks):
#         print(m)
#         print(d)
#         print()