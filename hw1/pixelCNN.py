import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from utils import tf_log_to_base_n

def get_pixelcnn_mask(kernel_size, in_channels, out_channels, isTypeA, n_channels=3):
    """
    raster ordering on conditioning mask

    kernel_size: size N of filter N x N
    in_channels: number of channels in
    out_channels: number of channels out
    isTypeA: bool, true if type A mask, otherwise type B mask used.
        Type A takes context and previous channels (but not its own channel)
        Type B takes context, prev channels and connected to own channel.

    We group the filters so that different filters correspond to different channels.
    first 3 are R, next 3 are G, last 3 are B

    Returns mask of shape (kernel_size, kernel_size, # channels, # filters)
    """
    mask = np.ones((kernel_size, kernel_size, in_channels, out_channels))
    centre = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if (j > centre) or (j == centre and i > centre):
                mask[j, i, :, :] = 0.
    for i in range(n_channels):
        for j in range(n_channels):
            if (isTypeA and i >= j) or (not isTypeA and i > j):
                mask[
                centre,
                centre,
                j::n_channels,
                i::n_channels,
                ] = 0.
    return mask


class MaskedCNN(tf.keras.layers.Conv2D):
    def __init__(self, n_filters, kernel_size, isTypeA, activation=None, **kwargs):
        """
        n_filters and kernel_size for conv layer
        isTypeA for mask type
        """
        assert isinstance(kernel_size, int), "Masked CNN requires square n x n kernel"
        super(MaskedCNN, self).__init__(n_filters, kernel_size, padding="SAME",
                                        activation=activation, **kwargs)
        self.isTypeA = isTypeA

    def build(self, input_shape):
        super().build(input_shape)
        (_, _, in_channels, out_channels) = self.kernel.shape
        self.mask = get_pixelcnn_mask(self.kernel_size[0], in_channels, out_channels, self.isTypeA)

    def call(self, inputs):
        # mask kernel for internal conv op, but then return to copy of kernel after for learning
        kernel_copy = self.kernel
        self.kernel = self.kernel * self.mask
        out = super().call(inputs)
        self.kernel = kernel_copy
        return out


class MaskedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters

    def build(self, input_shape):
        # 1x1 relu filter, then 3x3 then 1x1
        self.layer1 = MaskedCNN(self.n_filters, 1, False)
        self.layer2 = MaskedCNN(self.n_filters, 3, False)
        self.layer3 = MaskedCNN(self.n_filters*2, 1, False)

    def call(self, inputs):
        """
        x is the inputs, [image, (cx, cy)]
        """
        # other layers take img and cur pixel location
        x = tf.keras.layers.ReLU()(inputs)
        x = self.layer1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.layer2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.layer3(x)
        return inputs + x


class PixelCNNModel(tf.keras.Model):
    """
    Returns logits for softmax (N, h*w, c, n_vals)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_vals = 4
        self.output_channels = 3
        self.n_filters = 128

    def build(self, input_shape, **kwargs):
        self.layer1 = MaskedCNN(self.n_filters*2, 7, True)
        self.res_layers = [MaskedResidualBlock(self.n_filters) for _ in range(12)]
        # want ReLU applied first as per paper
        self.relu_conv1x1 = [tf.keras.layers.ReLU(),
                             MaskedCNN(self.n_filters, 1, False),
                             tf.keras.layers.ReLU(),
                             MaskedCNN(self.n_filters, 1, False)]
        self.output_conv = MaskedCNN(self.n_vals * self.output_channels, 1, False)

    def call(self, inputs, training=None, mask=None):
        img = tf.cast(inputs, tf.float32)
        x = self.layer1(img)
        for layer in self.res_layers:
            x = layer(x)
        for layer in self.relu_conv1x1:
            x = layer(x)
        x = self.output_conv(x)
        # output layer softmax split into n_channels
        n, h, w, _ = tf.shape(inputs)
        x = tf.reshape(x, (n, h,  w, self.output_channels, self.n_vals))
        return x


# eval, eval_batch, train_step, forward, loss, sample
class PixelCNN:
    def __init__(self, H=28, W=28, C=3, n_vals=4):
        """
        H, W, C image shape: height, width, channels
        n_vals the number of values each channel can take on
        """
        self.name = "PixelCNN"
        self.optimizer = tf.optimizers.Adam(learning_rate=10e-3)
        self.H = H
        self.W = W
        self.C = C
        self.n_vals = n_vals
        self.setup_model()

    def setup_model(self):
        self.model = PixelCNNModel()

    def loss(self, labels, logits):
        """
        probs are outputs of forward model, a probability for each image (N, )
        Returns mean *negative* log prob (likelihood) over x (a scalar)
        Autoregressive over space, ie. decomposes into a product over pixels conditioned on previous ones in ordering
        Since single dimension predicted each forward pass, logprob (base 2) is in bits per dimension
        """
        labels = tf.cast(labels, tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # get in bits
        neg_logprob_bit = tf_log_to_base_n(loss, 2)
        return neg_logprob_bit

    def forward_logits(self, x):
        """
        Forward pass returning full (flat) logits from model (N, H * W, C, N_V)
        where N_V is number of values each channel can take.
        """
        x = tf.cast(x, tf.int32)
        logits = self.model(x)
        return logits

    def forward_softmax(self, x):
        """
        Fwd pass retuning softmax values in image shape (N, H, W, C, N_V)
        """
        logits = self.forward_logits(x)
        probs = tf.nn.softmax(logits, axis=-1)
        return probs

    def train_step(self, X_train):
        """
        Takes batch of data X_train
        """
        with tf.GradientTape() as tape:
            logprob = self.fwd_loss(X_train)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob

    def eval_batch(self, X, bs=128):
        """
        computes forward pass then logprob on the outputs
        X is batched in to handle large data
        note this returns mean logprob over batch
        """
        neg_logprobs_bits = []
        for i in range(len(X) // bs):
            neg_logprobs_bits.append(self.fwd_loss(X[i * bs: (i + 1) * bs]))
        extra_data = X[len(X) // bs:]
        if len(extra_data) == 1:
            # add batch dimension
            extra_data = [extra_data]
        neg_logprobs_bits.append(self.fwd_loss(extra_data))
        return tf.reduce_mean(neg_logprobs_bits)

    def fwd_loss(self, X):
        X = tf.reshape(X, (-1, self.H, self.W, self.C))
        logits = self.forward_logits(X)
        loss = self.loss(X, logits)
        return loss

    def get_samples(self, n):
        """
        Generation is done from blank image (all 0s), we then sample R channel
        of first pixel, then G then B and then for second pixel etc.
        We batch this for efficiency.
        """
        images = np.zeros((n, self.H, self.W, self.C))
        # start with random values for first channel of first pixel
        images[:, 0, 0, 0] = np.random.choice(self.n_vals, size=n)
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C):
                    model_preds = self.forward_softmax(images)
                    # categorical over pixel values
                    pixel_dist = tfp.distributions.Categorical(probs=model_preds[:, h, w, c])
                    images[:, h, w, c] = pixel_dist.sample(1)
        return images


def plot_image(image, title, n_vals=3):
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255. / n_vals).astype(np.uint8), cmap="gray")
    if title is not None:
        plt.title(title)
        plt.savefig("figures/1_3/{}".format(title))
    plt.draw()
    plt.pause(0.001)


def display_image_grid(data, title):
    """
    data is shape (n, h, w, c)
    nd is sqrt(n) t0 make grid
    plots nd x nd grid of images
    """
    n = data.shape[0]
    h = data.shape[1]
    w = data.shape[2]
    c = data.shape[3]
    n_display = int(np.ceil(np.sqrt(n)))
    disp = np.zeros((h * n_display, w * n_display, c))
    for i in range(n_display):
        for j in range(n_display):
            data_ind = n_display * i + j
            if data_ind >= n:
                break
            disp[i * h: (i + 1) * h, j * w: (j + 1) * w] = data[data_ind]
    title = "{}-grid".format(title) if title is not None else None
    if c == 1:
        disp = np.squeeze(disp)
    plot_image(disp, title)


def test_maskA():
    mask = get_pixelcnn_mask(5, 3, 3, True)
    print(mask.shape)
    display_mask(mask, "Example Mask A")


def display_mask(mask, title):
    # index by prev layer's channels then this layer's channels
    # so rows are prev layer's channels, cols are this layer's
    # concat by prev layer channels into row images, then this channel dim is each image in the row (cols)
    mask_disp = np.concatenate(mask.transpose([3, 1, 0, 2]), axis=0).transpose([2, 1, 0])[..., None]
    # data is shape (n, h, w, c)
    # plots n rows of images
    c = mask_disp.shape[3]
    disp = np.concatenate(mask_disp, axis=0)
    title = "{}-grid".format(title) if title is not None else None
    if c == 1:
        disp = np.squeeze(disp)
    letters = ["R", "G", "B"]
    for i, j in enumerate([2, 7, 12]):
        plt.text(j, 15.5, letters[i])
        plt.text(-1.5, j, letters[i])
    plt.tick_params(labelbottom=False, labelleft=False)
    plot_image(disp, title)


def test_maskB():
    mask = get_pixelcnn_mask(5, 3, 3, False)
    display_mask(mask, "Example Mask B")


if __name__ == "__main__":
    test_maskA()
    test_maskB()

# TODO: clean up and simplify