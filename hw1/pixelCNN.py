import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from utils import tf_log_to_base_n


def get_pixelcnn_mask(kernel_size, in_channels, out_channels, isTypeA, n_channels=3, factorised=True):
    """
    Masks are repeated in groups with modulo if channel in or out != n_channels so if
    5 channels then it's R, R, G, G, B etc.
    This is so that when reshaping to (H, W, #channels, #values) we get each channel's values aligning
    For RGB channel taking N values case it's R1, R2, ... RN, G1, G2, ... GN, B1, B2, ... BN which reshapes to
    [R1, R2, ... RN], [G1, G2, ... GN], [B1, B2, ... BN]

    raster ordering on conditioning mask.

    kernel_size: size N of filter N x N
    in_channels: number of channels/filters in
    out_channels: number of channels/filters out
    n_channels: number of channels for masking eg. 3 for RGB masks
    isTypeA: bool, true if type A mask, otherwise type B mask used.
        Type A takes context and previous channels (but not its own channel)
        Type B takes context, prev channels and connected to own channel.
    factorised: bool, if True then probabilities treated independently P(r)p(g)p(b)
        so mask type A all have centre off and B all have it on.
        Otherwise the full joint as in the paper are used p(r)p(g|r)p(b|r,g)
        and A and B masks are different for each channel to allow this.

    Returns masks of shape (kernel_size, kernel_size, # in channels, # out channels)
    """
    channel_masks = np.ones((kernel_size, kernel_size, n_channels, n_channels), dtype=np.int32)
    centre = kernel_size // 2
    # bottom rows 0s
    channel_masks[centre+1:, :, :, :] = 0
    # right of centre on centre row 0s
    channel_masks[centre:, centre+1:, :, :] = 0
    # deal with centre based on mask "way": factorised or full
    # rows are channels in prev layer, columns are channels in this layer
    if factorised:
        if isTypeA:
            channel_masks[centre, centre, :, :] = 0
    else:
        # centre depends on mask type A or B
        k = 0 if isTypeA else 1
        # reverse i and j to get RGB ordering (other way would be BGR)
        i, j = np.triu_indices(n_channels, k)
        channel_masks[centre, centre, j, i] = 0.

    # we use repeat not tile because this keeps the correct ordering we need
    tile_shape = (int(np.ceil(in_channels / n_channels)), int(np.ceil(out_channels / n_channels)))
    masks = np.repeat(channel_masks, tile_shape[0], axis=2)
    masks = np.repeat(masks, tile_shape[1], axis=3)
    # tile the masks to potentially more than needed, then retrieve the number of channels wanted
    return masks[:, :, :in_channels, :out_channels]


class MaskedCNN(tf.keras.layers.Conv2D):
    def __init__(self, n_filters, kernel_size, isTypeA, factorised, activation=None, **kwargs):
        """
        n_filters and kernel_size for conv layer
        isTypeA for mask type
        """
        assert isinstance(kernel_size, int), "Masked CNN requires square n x n kernel"
        super(MaskedCNN, self).__init__(n_filters, kernel_size, padding="SAME",
                                        activation=activation, **kwargs)
        self.isTypeA = isTypeA
        self.factorised = factorised

    def build(self, input_shape):
        super().build(input_shape)
        (_, _, in_channels, out_channels) = self.kernel.shape
        self.mask = get_pixelcnn_mask(self.kernel_size[0], in_channels, out_channels, self.isTypeA,
                                      factorised=self.factorised)

    def call(self, inputs):
        # mask kernel for internal conv op, but then return to copy of kernel after for learning
        kernel_copy = self.kernel
        self.kernel = self.kernel * self.mask
        out = super().call(inputs)
        self.kernel = kernel_copy
        return out


class MaskedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, factorised):
        super().__init__()
        self.n_filters = n_filters
        self.factorised = factorised

    def build(self, input_shape):
        # 1x1 relu filter, then 3x3 then 1x1
        self.layer1 = MaskedCNN(self.n_filters, 1, False, self.factorised)
        self.layer2 = MaskedCNN(self.n_filters, 3, False, self.factorised)
        self.layer3 = MaskedCNN(self.n_filters * 2, 1, False, self.factorised)

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

    def __init__(self, H, W, C, n_vals, factorised, flat=False, *args, **kwargs):
        """
        :param flat: whether to keep flat or reshape each value for each channel variable
            if true then (bs, H, W, C * N) otherwise reshapes into logits (bs, H, W, C, N)
        """
        super().__init__(*args, **kwargs)
        self.n_vals = n_vals
        self.H = H
        self.W = W
        self.C = C
        self.n_filters = 128
        self.flat = flat
        self.factorised = factorised

    def build(self, input_shape, **kwargs):
        self.layer1 = MaskedCNN(self.n_filters * 2, 7, True, self.factorised)
        self.res_layers = [MaskedResidualBlock(self.n_filters, self.factorised) for _ in range(12)]
        # want ReLU applied first as per paper
        self.relu_conv1x1 = [tf.keras.layers.ReLU(),
                             MaskedCNN(self.n_filters, 1, False, self.factorised)]
        self.output_conv = [tf.keras.layers.ReLU(),
                            MaskedCNN(self.n_vals * self.C, 1, False, self.factorised)]

    def call(self, inputs, training=None, mask=None):
        img = tf.cast(inputs, tf.float32)
        x = self.layer1(img)
        for layer in self.res_layers:
            x = layer(x)
        for layer in self.relu_conv1x1:
            x = layer(x)
        for layer in self.output_conv:
            x = layer(x)
        if not self.flat:
            x = tf.reshape(x, (-1, self.H, self.W, self.C, self.n_vals))
        return x


# eval, eval_batch, train_step, forward, loss, sample
class PixelCNN:
    def __init__(self, H=28, W=28, C=3, n_vals=4, learning_rate=10e-4, factorised=True):
        """
        H, W, C image shape: height, width, channels
        n_vals the number of values each channel can take on
        """
        self.name = "PixelCNN"
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.H = H
        self.W = W
        self.C = C
        self.n_vals = n_vals
        self.factorised = factorised
        self.learning_rate = learning_rate
        self.setup_model()

    def __str__(self):
        return "Name: {}\nFactorised: {}\nLearning rate: {}\n".format(self.name,
                                                                    self.factorised, self.learning_rate)

    def setup_model(self):
        self.model = PixelCNNModel(self.H, self.W, self.C, self.n_vals, self.factorised)

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

    @tf.function
    def forward_softmax(self, x):
        """
        Fwd pass retuning softmax values in image shape (N, H, W, C, N_V)
        """
        logits = self.forward_logits(x)
        # seems to be numerical precision errors using float32
        logits_64 = tf.cast(logits, tf.float64)
        probs = tf.nn.softmax(logits_64, axis=-1)
        return probs

    def train_step(self, X_train):
        """
        Takes batch of data X_train
        returns logprob numpy
        """
        with tf.GradientTape() as tape:
            logprob = self.eval(X_train)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob.numpy()

    def eval_dataset(self, X, bs=64):
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

    @tf.function
    def eval(self, X):
        """
        Runs forward pass and loss
        :param X: input images batch
        :return: loss tensor
        """
        X = tf.reshape(X, (-1, self.H, self.W, self.C))
        logits = self.forward_logits(X)
        loss = self.loss(X, logits)
        return loss

    def get_samples(self, n, seed=123):
        """
        Generation is done from blank image (all 0s), we then sample R channel
        of first pixel, then G then B and then for second pixel etc.
        We batch this for efficiency.
        """
        images = np.zeros((n, self.H, self.W, self.C))
        # start with random values for first channel of first pixel (this is updated in first pass)
        images[:, 0, 0, 0] = np.random.choice(self.n_vals, n)
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C):
                    model_preds = self.forward_softmax(images)
                    # categorical over pixel values
                    pixel_dist = tfp.distributions.Categorical(probs=model_preds[:, h, w, c])
                    images[:, h, w, c] = pixel_dist.sample(1, seed=seed)
        return images


def plot_image(image, dir_path, title, n_vals=3):
    plt.clf()
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255. / n_vals).astype(np.uint8), cmap="gray")
    if title is not None:
        plt.title(title)
        plt.savefig("{}/{}.svg".format(dir_path, title))
    plt.draw()
    plt.pause(0.001)


def display_image_grid(data, dir_path, title):
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
    plot_image(disp, dir_path, title)


def test_maskA():
    mask = get_pixelcnn_mask(5, 3, 3, True)
    print(mask.shape)
    display_mask(mask, "Example Mask A")


def display_mask(mask, title):
    # index by prev layer's channels then this layer's channels
    # so rows are prev layer's channels, cols are this layer's
    # concat by prev layer channels into row images, then this channel dim is each image in the row (cols)
    mask_disp = np.concatenate(mask.transpose([3, 1, 0, 2]), axis=0).transpose([2, 1, 0])
    # data is shape (n, h, w, c)
    # plots n rows of images
    disp = np.concatenate(mask_disp, axis=0)
    title = "{}-grid".format(title) if title is not None else None
    letters = ["R", "G", "B"]
    for i, j in enumerate([2, 7, 12]):
        plt.text(j, 15.5, letters[i])
        plt.text(-1.5, j, letters[i])
    plt.tick_params(labelbottom=False, labelleft=False)
    plot_image(disp, "figures/1_3", title)


def test_maskB():
    mask = get_pixelcnn_mask(5, 3, 3, False)
    display_mask(mask, "Example Mask B")


if __name__ == "__main__":
    # test_maskA()
    # test_maskB()

    kernel_size, inc, outc, typeA = 5, 3, 4, True
    mask = get_pixelcnn_mask(kernel_size, inc, outc, typeA)
    display_mask(mask, None)
