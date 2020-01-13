# Source:https://github.com/davidsandberg/unsupervised/blob/master/HW1/HW1_3.ipynb
# for comparing / debugging my code

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt
from pixelCNN import get_mask

def masked_conv2d(x, channels_out, kernel_size, input_channels, mask_type, factorized):
    # Get dimensions of the input tensor
    _, h, w, channels_in = x.shape.as_list()
    # Create weight and bias variables
    weights = tf.compat.v1.get_variable('weight', shape=(kernel_size, kernel_size, channels_in, channels_out), trainable=True)
    bias = tf.compat.v1.get_variable('bias', shape=(h, w, channels_out), trainable=True)
    # Create the mask
    mask = get_mask(kernel_size, channels_in, channels_out, input_channels=input_channels, mask_type=mask_type,
                    factorized=factorized)
    # Apply convolution
    y = tf.nn.conv2d(x, weights * mask, strides=[1, 1, 1, 1], padding='SAME') + bias
    return y


def res_block_with_bug(x, channels_out=128, input_channels=None, factorized=True):
    y = tf.nn.relu(x)
    # Downsample channels using 1x1 convolution
    with tf.compat.v1.variable_scope('downsample'):
        y = masked_conv2d(x, channels_out=channels_out, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    y = tf.nn.relu(x)
    # Main convolution
    with tf.compat.v1.variable_scope('conv'):
        y = masked_conv2d(x, channels_out=channels_out, kernel_size=3, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    y = tf.nn.relu(x)
    # Upsample channels by two using 1x1 convolution
    with tf.compat.v1.variable_scope('upsample'):
        y = masked_conv2d(x, channels_out=channels_out * 2, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    return y + x


def res_block(x_in, channels_out=128, input_channels=None, factorized=True):
    x = tf.nn.relu(x_in)
    # Downsample channels using 1x1 convolution
    with tf.compat.v1.variable_scope('downsample'):
        x = masked_conv2d(x, channels_out=channels_out, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    x = tf.nn.relu(x)
    # Main convolution
    with tf.compat.v1.variable_scope('conv'):
        x = masked_conv2d(x, channels_out=channels_out, kernel_size=3, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    x = tf.nn.relu(x)
    # Upsample channels by two using 1x1 convolution
    with tf.compat.v1.variable_scope('upsample'):
        x = masked_conv2d(x, channels_out=channels_out * 2, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    return x + x_in


class PixelCNNDS:
    def __init__(self, H=28, W=28, C=3, N=4, factorized=True, learning_rate=10e-3):
        self.name = "PixelCNN-DS"
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.factorized = factorized
        self.optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        self.sess = tf.compat.v1.Session()
        self.setup_model()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def setup_model(self):
        self.x_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 3))
        input_channels = self.x_ph.shape.as_list()[3]
        inp = tf.cast(self.x_ph, tf.int32)
        x = tf.cast(self.x_ph, tf.float32)
        with tf.compat.v1.variable_scope('input_conv'):
            x = masked_conv2d(x, channels_out=128 * 2, kernel_size=7, input_channels=input_channels, mask_type='A',
                                             factorized=self.factorized)
        for i in range(12):
            with tf.compat.v1.variable_scope('res_block_%d' % i):
                x = res_block(x, channels_out=128, input_channels=input_channels, factorized=self.factorized)
        with tf.compat.v1.variable_scope('output_conv_1'):
            x = tf.nn.relu(x)
            x = masked_conv2d(x, channels_out=128, kernel_size=1, input_channels=input_channels, mask_type='B',
                                             factorized=self.factorized)
        with tf.compat.v1.variable_scope('output_conv_2'):
            x = tf.nn.relu(x)
            x = masked_conv2d(x, channels_out=self.N * self.C, kernel_size=1, input_channels=input_channels, mask_type='B',
                                      factorized=self.factorized)

        x_rshp = tf.reshape(x, [-1, 28, 28, 3, 4])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inp, logits=x_rshp)
        self.loss = tf.reduce_mean(losses) * np.log2(np.e)
        self.probs = tf.nn.softmax(x_rshp)
        self.update_op = self.optimiser.minimize(self.loss, var_list=tf.compat.v1.trainable_variables())

    def train_step(self, X):
        loss, _ = self.sess.run([self.loss, self.update_op], feed_dict={self.x_ph: X})
        return loss

    def eval(self, X):
        return self.sess.run(self.loss, feed_dict={self.x_ph: X})

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
        mean_nll = np.mean(neg_logprobs_bits)
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
                    for i in range(n):
                        images[i, h, w, c] = np.random.choice(self.N, p=model_preds[i, h, w, c])
        return images

    def forward_softmax(self, X):
        return self.sess.run(self.probs, feed_dict={self.x_ph: X})


def compare_sampling(n=10, m=10000, seed=123):
    """
    Compares np.random.choice with tfp.distributions.Cateorgical().sample
    n - dimensional RVs
    m - samples
    """
    np.random.seed(seed)
    x = np.random.random(size=n)
    x = x / np.sum(x)
    y_np = np.random.choice(n, size=m, p=x)
    y_tfp = tfp.distributions.Categorical(probs=x).sample(m, seed=seed)
    freq_np = np.zeros(n)
    freq_tfp = np.zeros(n)
    for i in range(n):
        freq_np[i] = np.sum(y_np == i)
        freq_tfp[i] = np.sum(y_tfp == i)
    freq_np = freq_np / m
    freq_tfp = freq_tfp / m
    plt.bar(range(n), freq_np, label="NP", edgecolor="r", color="None")
    plt.bar(range(n), freq_tfp, label="TFP", edgecolor="g", color="None")
    plt.bar(range(n), x, label="True", edgecolor="b", color="None")
    plt.legend()
    plt.show()
    sns.distplot(freq_np, label="np")
    sns.distplot(freq_tfp, label="tfp")
    sns.distplot(x, label="true")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_sampling()