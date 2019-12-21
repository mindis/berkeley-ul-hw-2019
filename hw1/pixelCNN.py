import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from utils import tf_log2, gather_nd


def get_pixelcnn_mask(this_filters_shape, isTypeA, n_channels=3):
    """
    raster ordering on conditioning

    this_filters_shape define the layer sizes
    isTypeA: bool, true if type A mask, otherwise type B mask used.
    Type A takes context and previous channels (but not its own channel)
    Type B takes context, prev channels and connected to own channel.

    We group the filters so that different filters correspond to different channels.
    first 3 are R, next 3 are G, last 3 are B

    Returns mask of shape (kernel_size, kernel_size, # channels, # filters)
    """
    mask = np.ones(this_filters_shape)
    ctr_pix = [this_filters_shape[0] // 2, this_filters_shape[1] // 2]
    # mask out all pixels conditioned on
    for i in range(this_filters_shape[0]):
        for j in range(this_filters_shape[1]):
            # deal with centre pixel, then by ordering the other pixels are all 1
            if i == ctr_pix[0] and j == ctr_pix[1]:
                mask_centre(ctr_pix, isTypeA, mask, n_channels)
                break
            # mask all channels for conditioned pixels
            # TODO: should this be other way? due to image y axis flipped
            elif i < ctr_pix[0] or (i == ctr_pix[0] and j < ctr_pix[1]):
                mask[i, j, :, :] = 0
    return mask


def mask_centre(cur_pixel, isTypeA, mask, n_channels):
    """
    cur_pixel location (row, col)
    mask is the mask to build of shape filters_shape
    n_channels is the number of channels usually 3

    first 3 are R, next 3 are G, last 3 are B
    mask of shape (kernel_size, kernel_size, # channels, # filters)
    so 3rd dimension (# channels) is prev layer's RGB
    and 4th dimension (# filters) is next layer's RGB
    """
    for i in range(n_channels):
        for j in range(n_channels):
            if isTypeA and i >= j:
                mask[cur_pixel[0], cur_pixel[1], i::n_channels, j::n_channels] = 0
            elif i > j:
                mask[cur_pixel[0], cur_pixel[1], i::n_channels, j::n_channels] = 0


class MaskedCNN(tf.keras.layers.Conv2D):
    def __init__(self, n_filters, kernel_size, isTypeA, activation=None, **kwargs):
        """
        n_filters and kernel_size for conv layer
        isTypeA for mask type
        """
        super(MaskedCNN, self).__init__(n_filters, kernel_size, padding="SAME",
                                        activation=activation, **kwargs)
        self.isTypeA = isTypeA

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = get_pixelcnn_mask(self.kernel.shape, self.isTypeA)

    def call(self, inputs):
        self.kernel = self.kernel * self.mask
        return super().call(inputs)


class MaskedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters

    def build(self, input_shape):
        # 1x1 relu filter, then 3x3 then 1x1
        self.layer1 = MaskedCNN(self.n_filters, 1, False, activation="relu")
        self.layer2 = MaskedCNN(self.n_filters, 3, False, activation="relu")
        self.layer3 = MaskedCNN(self.n_filters*2, 1, False, activation="relu")

    def call(self, inputs):
        """
        x is the inputs, [image, (cx, cy)]
        """
        # other layers take img and cur pixel location
        x = self.layer1(inputs)
        x = self.layer2(x)
        res_path = self.layer3(x)
        return inputs + res_path


class PixelCNNModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_size = 4
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
        self.output_conv = MaskedCNN(self.output_size * self.output_channels, 1, False)
        self.softmax = tf.keras.layers.Softmax()

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
        x = self.softmax(tf.reshape(x, (n, h * w, self.output_channels, self.output_size)))
        return x


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

    def sum_logprob(self, probs):
        """
        probs are outputs of forward model
        Returns mean *negative* log prob (likelihood) over x (a scalar)
        Autoregressive over space, ie. decomposes into a product over pixels conditioned on previous ones in ordering
        Here we further assume that it factorizes into product over channels
        """
        # factorised over channels (multiply) and over masked conditioned input.
        # in log space this is sum. We also sum logprob over examples, but divid by number of examples to get mean
        # we also divide by number of dimensions to get bits per dimension
        n_examples = tf.cast(tf.shape(probs)[0], tf.float32)
        n_dimensions = tf.cast(self.H * self.W * self.C, tf.float32)
        logprob = tf.reduce_sum(tf_log2(probs)) / (n_examples * n_dimensions)
        return -logprob

    def forward_gather(self, x):
        """
        Forward pass returning (N, H, W, C) where elements are probs of
        corresponding input value.
        """
        x = tf.cast(x, tf.int32)
        model_outputs = self.forward(x)
        # we flatten the softmax, gather the corresponding probs to the input values for each channel input in x.
        # flatten to (h x w x c, n_vals (=4)) then softmax then flatten image to loss.
        N = tf.shape(x)[0]
        flat_size = N * self.H * self.W * self.C
        probs_flat = gather_nd(tf.reshape(model_outputs, (flat_size, self.n_vals)), tf.reshape(x, (flat_size,)))
        # reshape back to image shape for probs
        probs = tf.reshape(probs_flat, (N, self.H, self.W, self.C))
        return probs

    def forward(self, x):
        """
        Forward pass returning full (flat) probability values (N, H * W * C, N_V)
        where N_V is number of values each channel can take. ie this gives the
        output of the softmax
        """
        # prob is output for the unit corresponding to value of this pixel
        x = tf.cast(x, tf.int32)
        # model outputs softmax (N, H * W * C, N_V) where N is number of data in batch, Height, Width, Channels of image
        # and N_V is number of values each channel can take
        model_outputs = self.model(x)
        return model_outputs

    def train_step(self, X_train):
        """
        Takes batch of data X_train
        """
        with tf.GradientTape() as tape:
            # TODO: loss as tf softmax x_ent w logits? clip grads?
            logprob = self.eval(X_train)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob

    def eval_batch(self, X, bs=128):
        """
        computes forward pass then logprob on the outputs
        X is batched in to handle large data
        note this returns mean logprob over batch
        """
        logprobs = []
        for i in range(len(X) // bs):
            logprobs.append(self.eval(X[i * bs: (i + 1) * bs]))
        extra_data = X[len(X) // bs:]
        if len(extra_data) == 1:
            # add batch dimension
            extra_data = [extra_data]
        logprobs.append(self.eval(extra_data))
        return tf.reduce_mean(logprobs)

    def eval(self, X):
        preds = self.forward_gather(X)
        logprob = self.sum_logprob(preds)
        return logprob

    def reshape_model_outputs(self, probs):
        """
        Reshapes softmax output (N, H * W * C, N_V) to image shape
        (N, H, W, C, N_V) probs
        """
        N = tf.shape(probs)[0]
        return tf.reshape(probs, (N, self.H, self.W, self.C, self.n_vals))

    def get_samples(self, n):
        """
        Generation is done from blank image (all 0s), we then sample R channel
        of first pixel, then G then B and then for second pixel etc.
        We batch this for efficiency.
        """
        images = np.zeros((n, self.H, self.W, self.C))
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C):
                    model_preds_flat = self.forward(images)
                    model_preds = tf.reshape(model_preds_flat, (n, self.H, self.W, self.C, self.n_vals))
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


def display_image_rows(data, title):
    """
    data is shape (n, h, w, c)
    plots n rows of images
    """
    c = data.shape[3]
    disp = np.concatenate(data, axis=0)
    title = "{}-grid".format(title) if title is not None else None
    if c == 1:
        disp = np.squeeze(disp)
    plot_image(disp, title)


def test_maskA():
    mask_shape = (5, 5, 3, 3)
    mask = get_pixelcnn_mask(mask_shape, True)
    # index by prev layer's channels then this layer's channels
    # so rows are prev layer's channels, cols are this layer's
    # concat by prev layer channels into row images, then this channel dim is each image in the row (cols)
    mask_disp = np.concatenate(mask.transpose([3, 1, 0, 2]), axis=0).transpose([2, 1, 0])[..., None]
    print(mask_disp.shape)
    # mask_disp = mask.reshape(mask_shape[:2] + (-1, 1)).transpose([2, 0, 1, 3])
    display_image_rows(mask_disp, "Example Mask A")


def test_maskB():
    mask_shape = (5, 5, 3, 3)
    mask = get_pixelcnn_mask(mask_shape, False)
    # index by prev layer's channels then this layer's channels
    # so rows are prev layer's channels, cols are this layer's
    # concat by prev layer channels into row images, then this channel dim is each image in the row (cols)
    mask_disp = np.concatenate(mask.transpose([3, 1, 0, 2]), axis=0).transpose([2, 1, 0])[..., None]
    print(mask_disp.shape)
    # mask_disp = mask.reshape(mask_shape[:2] + (-1, 1)).transpose([2, 0, 1, 3])
    display_image_rows(mask_disp, "Example Mask B")


if __name__ == "__main__":
    test_maskA()
    test_maskB()
