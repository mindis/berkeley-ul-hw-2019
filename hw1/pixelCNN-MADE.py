import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from utils import tf_log_to_base_n

# eval, eval_batch, train_step, forward, loss, sample
class PixelCNN:
    def __init__(self, H=28, W=28, C=3, n_vals=4, learning_rate=10e-4):
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
        self.setup_model()

    def setup_model(self):
        # TODO: make sure uses factorised?
        pixelcnn = PixelCNNModel()
        flatten = Flatten()(pixelcnn)
        # TODO: rewrited MADE layer to accept different input size to output?
        #   need 3 x 4 output, but want 28 * 28 * 3 * 4 input? or narrow it down with a dense layer?
        self.model = MADE

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
        # for small batch just eval it
        if len(X) <= bs:
            return self.eval(X)
        # otherwise evaluate in batches
        neg_logprobs_bits = []
        # eval in batches
        for i in range(len(X) // bs):
            neg_logprobs_bits.append(self.eval(X[i * bs: (i + 1) * bs]))
        # mean of batches
        mean_nll = tf.reduce_mean(neg_logprobs_bits)
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

    def eval(self, X):
        """
        Runs forward pass and loss
        :param X: input images batch
        :return: loss
        """
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
        # start with random values for first channel of first pixel (this is updated in first pass)
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
    plt.clf()
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255. / n_vals).astype(np.uint8), cmap="gray")
    if title is not None:
        plt.title(title)
        plt.savefig("figures/1_3/{}.svg".format(title))
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
    # test_maskA()
    # test_maskB()

    kernel_size, inc, outc, typeA = 5, 3, 4, True
    mask = get_pixelcnn_mask(kernel_size, inc, outc, typeA)
    display_mask(mask, None)
