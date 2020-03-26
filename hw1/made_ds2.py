import os

import tensorflow as tf
import tensorflow_probability as tfpZ
import numpy as np
import matplotlib.pyplot as plt


def mask(size, type_A):
    m = np.zeros((size, size), dtype=np.float32)
    m[:size // 2, :] = 1
    m[size // 2, :size // 2] = 1
    if not type_A:
        m[size // 2, size // 2] = 1
    return m


class ConvMasked(tf.keras.layers.Layer):
    def __init__(self, name, size, in_channels=128, out_channels=128, type_A=False, **kwargs):
        super().__init__(name, **kwargs)
        self._name = name
        self._size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.type_A = type_A

    def build(self, input_shape, **kwargs):
        self.conv_filter = tf.Variable(tf.initializers.glorot_normal()((self._size, self._size, self.in_channels, self.out_channels)),
            name=self._name + '_filter', trainable=True)
        self.conv_bias = tf.Variable(tf.initializers.glorot_normal()((input_shape[1], input_shape[2], self.out_channels)),
                                     name=self._name + '_bias',
                                    trainable=True)

    def call(self, inputs, **kwargs):
        masked_conv_filter = self.conv_filter * mask(self._size, self.type_A)[:, :, np.newaxis, np.newaxis]
        return tf.nn.conv2d(inputs, masked_conv_filter, strides=[1, 1, 1, 1], padding='SAME', name=self._name) \
               + self.conv_bias


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, scope, channels=128, **kwargs):
        super().__init__(**kwargs)
        self.scope = scope
        self._channels = channels

    def build(self, input_shape, **kwargs):
        self._layers = []
        self._layers.append(tf.keras.activations.relu)
        self._layers.append(ConvMasked('conv1x1_downsample', 1, self._channels, self._channels // 2))
        self._layers.append(tf.keras.activations.relu)
        self._layers.append(ConvMasked('conv3x3', 3, self._channels // 2, self._channels // 2))
        self._layers.append(tf.keras.activations.relu)
        self._layers.append(ConvMasked('conv1x1_upsample', 1, self._channels // 2, self._channels))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return inputs + x


class PixelCNN(tf.keras.layers.Layer):
    def __init__(self, channels, final_channels, **kwargs):
        super().__init__(**kwargs)
        self._layers = []
        self._layers.append(ConvMasked(name='start_conv7x7', size=7, in_channels=3,
                          out_channels=channels, type_A=True))

        for i in range(12):
            self._layers.append(ResBlock(scope='res_block_{}'.format(i+1),
                            channels=channels))

        self._layers.append(ConvMasked('final_conv3x3', size=3,
                          in_channels=channels,
                          out_channels=channels))
        self._layers.append(tf.keras.activations.relu)
        self._layers.append(ConvMasked('final_conv1x1_1', 1,
                       in_channels=channels,
                       out_channels=channels))
        self._layers.append(tf.keras.activations.relu)
        self._layers.append(ConvMasked('final_conv1x1_2', 1,
                       in_channels=channels,
                       out_channels=final_channels))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

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
        self.D = input_shape[3]

    def call(self, inputs, **kwargs):
        units = tf.reshape(
            tf.one_hot(tf.cast(inputs, tf.int32), depth=self.depth, dtype=tf.float32),
                shape = (tf.shape(inputs)[0],
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
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.weight_mask = np.ones((self.unit_count,
                                    input_shape[-1]),
                                   dtype=np.bool)
        self.weight_mask[:self.m.shape[-1],
        :self.prev_units.shape[-1]] = self.m[:, np.newaxis] >= self.prev_units[np.newaxis, :]
        self.W = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                             self.height,
                                             self.unit_count,
                                             input_shape[-1])), name="W")
        self.b = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                        self.height,
                                        self.unit_count)), name="b")

    def call(self, inputs, **kwargs):
        # expects input to be prev unit || aux
        x = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask, inputs) + self.b
        x = tf.nn.relu(x)

        return x

class MadeHidden(tf.keras.layers.Layer):
    def __init__(self, made_prev_layer_units, D, depth, unit_count, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.depth = depth
        self.prev_units = made_prev_layer_units
        self.unit_count = unit_count
        self.m = np.mod(np.arange(self.unit_count), self.D - 1)

    def build(self, input_shape):
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.weight_mask = self.m[:, np.newaxis] >= self.prev_units[np.newaxis, :]
        self.W = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                              self.height,
                                                              self.unit_count,
                                                              input_shape[-1])), name="W")
        self.b = tf.Variable(tf.initializers.glorot_normal()((self.width,
                                                              self.height,
                                                              self.unit_count)), name="b")

    def call(self, inputs, **kwargs):
        x = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask, inputs) + self.b
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
        self.width = input_shape[0][1]
        self.height = input_shape[0][2]
        self.weight_mask = self.m[:, np.newaxis] > self.prev_units[np.newaxis, :]
        self.W = tf.compat.v1.get_variable("W", shape=(self.width,
                                                       self.height,
                                                       self.D * self.depth,
                                                       input_shape[0][-1]))
        self.b = tf.compat.v1.get_variable("b", shape=(self.width,
                                                       self.height,
                                                       self.D * self.depth))

        self.direct_mask = np.repeat(np.tril(np.ones(self.D), -1), self.depth).reshape((self.D, -1))
        self.direct_mask = np.repeat(self.direct_mask, self.depth, axis=0)
        self.A = tf.compat.v1.get_variable("A", shape=(self.width,
                                                       self.height,
                                                       self.D * self.depth,
                                                       self.D * self.depth))

    def call(self, inputs, **kwargs):
        # expects input (inputs, aux)
        inputs, aux = inputs
        self.units = tf.einsum('whij,bwhj->bwhi',
                               self.W * self.weight_mask,
                               inputs) + self.b
        # self.units += tf.einsum('whij,bwhj->bwhi',
        #                         self.A * self.direct_mask,
        #                         res_from_first_layer)

        self.unconnected_W = tf.compat.v1.get_variable('unconnected_W',
                                             shape=(self.width,
                                                    self.height,
                                                    self.depth,
                                                    aux.shape[-1]))
        self.unconnected_b = tf.compat.v1.get_variable('unconnected_b',
                                             shape=(self.width,
                                                    self.height,
                                                    self.depth))
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
        self.pixelCNN = PixelCNN(128, 16)
        self.Made_layers = []
        self.Made_layers.append(MadeInput(3, 4))
        self.Made_layers.append(MadeHiddenWithAuxiliary(self.Made_layers[-1].m, 3, 4, 32))
        self.Made_layers.append(MadeOutput(self.Made_layers[0].m, self.Made_layers[-1].m, 3, 4))

    def call(self, inputs, training=None, mask=None):
        x = tf.cast(inputs, tf.float32) * 1./3.
        x = self.pixelCNN(x)
        aux = tf.keras.activations.relu(x)
        x = aux
        for layer in self.Made_layers[:-1]:
            x = layer(x)
        x = self.Made_layers[-1]((x, aux))
        return x


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
