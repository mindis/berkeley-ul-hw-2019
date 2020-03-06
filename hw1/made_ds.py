import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from pixelCNN import PixelCNN


def dense_masked(x, nrof_units, mask, activation=None):
    nrof_inputs = x.get_shape()[1]
    kernel = tf.Variable(shape=(nrof_inputs, nrof_units), dtype=tf.float32,
                             initializer=tf.initializers.glorot_normal, trainable=True)
    bias = tf.Variable(shape=(nrof_units,), dtype=tf.float32,
                           initializer=tf.initializers.zeros, trainable=True)
    y = tf.tensordot(x, kernel * mask, 1) + bias
    y = activation(y) if activation else y
    return y


def get_masks(nrof_units, nrof_layers, nrof_dims, nrof_aux, nrof_bins):
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


def made(x, aux, nrof_units, nrof_layers, nrof_dims, nrof_aux, nrof_bins):
    x = tf.cast(x, tf.float32)
    masks = get_masks(nrof_units, nrof_layers, nrof_dims, nrof_aux, nrof_bins)
    hidden = [nrof_units] * nrof_layers + [nrof_dims * nrof_bins]
    for i, h in enumerate(hidden):
        activation = tf.nn.relu if i < nrof_layers else None
        xc = tf.concat([aux, x], -1)
        x = dense_masked(xc, h, masks[i], activation=activation)
    return x


class DS_PixelCNN_MADE_Model(tf.keras.Model):
    def __init__(self):
        self.pixelCNN = PixelCNN()

    def call(self, inputs, training=None, mask=None):
        x_made = tf.reshape(tf.one_hot(inputs, depth=4), (-1, 3 * 4))
        aux = self.pixelCNN(inputs)
        aux_rshp = tf.reshape(aux, (-1, 3 * 4))
        y_made = made(x_made, aux_rshp, nrof_units=128, nrof_layers=2, nrof_dims=3, nrof_aux=3 * 4, nrof_bins=4)
        y_made_rshp = tf.reshape(y_made, (-1, 28, 28, 3, 4))
        return y_made_rshp


class DS_PixelCNN_MADE:
    def __init__(self, lr=10e-4):
        self.H, self.W, self.C = 28, 28, 3
        self.N = 4
        self.model = DS_PixelCNN_MADE_Model()
        self.optimizer = tf.optimizers.Adam(lr)

    def eval(self, x):
        y = self.model(x)
        return self.loss(x, y).numpy()

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

