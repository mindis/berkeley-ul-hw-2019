import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from pixelCNN import PixelCNNModel


class MadeInput(tf.keras.layers.Layer):
    def __init__(self, D, depth, **kwargs):
            super().__init__(**kwargs)
            self.D = D
            self.depth = depth
            self.m = np.arange(self.D)
            self.m = np.repeat(self.m, self.depth, axis=-1)

    def build(self, input_shape):
        self.width = input_shape[1]
        self.height = input_shape[2]

    def call(self, inputs, **kwargs):
        units = tf.reshape(
            tf.one_hot(tf.cast(inputs, tf.int32), depth=self.depth, dtype=tf.float32),
                shape = (-1,
                         self.width,
                         self.height,
                         self.depth * self.D))
        return units


class MadeHiddenWithAuxiliary(tf.keras.layers.Layer):
    def __init__(self, made_prev_layer_units, D, depth, unit_count, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.depth = depth
        self.prev_units = made_prev_layer_units
        self.unit_count = unit_count
        self.m = np.mod(np.arange(self.unit_count), self.D - 1)

    def build(self, input_shape):
        self.width = input_shape[0][1]
        self.height = input_shape[0][2]
        shape_ = input_shape[0][-1] + input_shape[1][-1]
        self.weight_mask = np.ones((self.unit_count,
                                    shape_),
                                   dtype=np.bool)
        self.weight_mask[:self.m.shape[-1],
        :self.prev_units.shape[-1]] = self.m[:, np.newaxis] >= self.prev_units[np.newaxis, :]
        self.W = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                             self.height,
                                             self.unit_count,
                                                              shape_)), name="W")
        self.b = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                        self.height,
                                        self.unit_count)), name="b")

    def call(self, inputs, **kwargs):
        # expects input to be (prev unit, aux)
        inputs = tf.concat(inputs, -1)
        tf.print(tf.shape(inputs))
        tf.print(tf.shape(self.W))
        x = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask, inputs) + self.b
        tf.print(tf.shape(x))
        tf.print("")
        x = tf.nn.relu(x)
        return x

class MadeOutput(tf.keras.layers.Layer):
    def __init__(self, in_units, made_prev_layer_units, D, depth, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.depth = depth
        self.in_units = in_units
        self.prev_units = made_prev_layer_units
        self.m = np.repeat(np.arange(self.D), self.depth)

    def build(self, input_shape):
        self.width = input_shape[1][1]
        self.height = input_shape[1][2]
        self.weight_mask = self.m[:, np.newaxis] > self.prev_units[np.newaxis, :]
        self.W = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                       self.height,
                                                       self.D * self.depth,
                                                       input_shape[1][-1])))
        self.b = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                       self.height,
                                                       self.D * self.depth)))

        self.direct_mask = np.repeat(np.tril(np.ones(self.D), -1), self.depth).reshape((self.D, -1))
        self.direct_mask = np.repeat(self.direct_mask, self.depth, axis=0)
        self.A = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                       self.height,
                                                       self.D * self.depth,
                                                       self.D * self.depth)))
        self.unconnected_W = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                    self.height,
                                                    self.depth,
                                                    input_shape[2][-1])))
        self.unconnected_b = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                    self.height,
                                                    self.depth)))
    def call(self, inputs, **kwargs):
        # expects input (res, inputs, aux)
        res, inputs, aux = inputs
        self.units = tf.einsum('whij,bwhj->bwhi',
                               self.W * self.weight_mask,
                               inputs) + self.b
        self.units += tf.einsum('whij,bwhj->bwhi',
                                self.A * self.direct_mask,
                                res)

        self.unconnected_out = tf.einsum('whij,bwhj->bwhi',
                                         self.unconnected_W,
                                         aux) + self.unconnected_b
        self.units = tf.concat([self.unconnected_out,
                                self.units[:, :, :, self.depth:]],
                               axis=3)
        self.units = tf.reshape(self.units, shape=(-1,
                                                   self.width,
                                                   self.height,
                                                   self.D,
                                                   self.depth))
        return self.units


class DS_PixelCNN_MADE_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.pixelCNN = PixelCNNModel(28, 28, 3, 4, True, True)
        self.made_in = MadeInput(3, 4)
        self.made_h = MadeHiddenWithAuxiliary(self.made_in.m, 3, 4, 32)
        self.made_out = MadeOutput(self.made_in.m, self.made_h.m, 3, 4)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        x = self.pixelCNN(inputs * 1./3.)
        pixel_cnn_out = tf.nn.relu(x)
        made_in = self.made_in(inputs)
        made_h = self.made_h((made_in, pixel_cnn_out))
        made_out = self.made_out((made_in, made_h, pixel_cnn_out))
        return made_out


class DS_PixelCNN_MADE:
    def __init__(self, H=28, W=28, C=3, N=4, D=3, learning_rate=10e-4, n_hidden_units=124):
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.D = D
        self.name = "DS_PixelCNN_MADE"
        self.n_hidden_units = n_hidden_units
        self.model = DS_PixelCNN_MADE_Model()
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
        grads, _ = tf.clip_by_global_norm(grads, 5)
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


if __name__ == "__main__":
    # W is width, height, units, d * depth whij # (2,2,3,4)
    # x is bs, width, height, d * depth bwhj # (5,2,2,4)
    # out is bs, width, height, units # (5,2,2,3)
    # W = np.repeat(np.arange(4), 2 * 2 * 3).reshape((2,2,3,4))
    # W = np.tile(np.arange(4*3), 2 * 2).reshape((2,2,3,4))
    W = np.tile(np.arange(3*4*2, dtype=np.float), 2).reshape((2,2,4,3)).transpose((0,1,3,2))
    # print(W)
    x = np.ones((5,2,2,4))
    print(tf.einsum('whij,bwhj->bwhi', W, x))
    print(tf.reshape(tf.matmul(W.reshape(4*2, 2*3).T, x.reshape(5 * 2, 2*4).T), (5,2,2,3)))