# Source:https://github.com/davidsandberg/unsupervised/blob/master/HW1/HW1_3.ipynb
# for comparing / debugging my code

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt

from pixelCNN import get_pixelcnn_mask, display_mask, plot_image, display_image_grid


def get_mask(kernel_size, channels_in, channels_out, input_channels, mask_type, factorized=True):
    mask = np.zeros(shape=(kernel_size, kernel_size, channels_in, channels_out), dtype=np.float32)
    mask[:kernel_size // 2, :, :, :] = 1
    mask[kernel_size // 2, :kernel_size // 2, :, :] = 1

    if factorized:
        if mask_type == 'B':
            mask[kernel_size // 2, kernel_size // 2, :, :] = 1
    else:
        factor_w = int(np.ceil(channels_out / input_channels))
        factor_h = int(np.ceil(channels_in / input_channels))
        k = mask_type == 'A'
        m0 = np.triu(np.ones(dtype=np.float32, shape=(input_channels, input_channels)), k)
        m1 = np.repeat(m0, factor_w, axis=1)
        m2 = np.repeat(m1, factor_h, axis=0)
        mask_ch = m2[:channels_in, :channels_out]
        mask[kernel_size // 2, kernel_size // 2, :, :] = mask_ch

    return mask


class MaskedConv2d(tf.keras.layers.Layer):
    def __init__(self, channels_out, kernel_size, input_channels, mask_type, factorized):
        super().__init__()
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.mask_type = mask_type
        self._factorised = factorized

    def build(self, input_shape):
        # Get dimensions of the input tensor
        _, h, w, channels_in = input_shape
        # 1x1 relu filter, then 3x3 then 1x1
        # Create weight and bias variables
        shape = (self.kernel_size, self.kernel_size, channels_in, self.channels_out)
        self._weights = tf.Variable(tf.initializers.glorot_uniform()(shape), name='weight', shape=shape,
                              trainable=True)
        shape_bias = (h, w, self.channels_out)
        self._bias = tf.Variable(tf.initializers.glorot_uniform()(shape_bias), name='bias', shape=shape_bias, trainable=True)
        # Create the mask
        self._mask = get_mask(self.kernel_size, channels_in, self.channels_out, input_channels=self.input_channels, mask_type=self.mask_type,
                        factorized=self._factorised)

    def call(self, x):
        # Apply convolution
        y = tf.nn.conv2d(x, self._weights * self._mask, strides=[1, 1, 1, 1], padding='SAME') + self._bias
        return y


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_out, input_channels, factorized=True):
        super().__init__()
        self.channels_out = channels_out
        self.input_channels = input_channels
        self.factorized = factorized

    def build(self, input_shape):
        self._layers = []
        self._layers.append(tf.keras.layers.Activation("relu"))
        # Downsample channels using 1x1 convolution
        self._layers.append(MaskedConv2d(channels_out=self.channels_out, kernel_size=1, input_channels=self.input_channels, mask_type='B',
                                         factorized=self.factorized))
        self._layers.append(tf.keras.layers.Activation("relu"))
        # Main convolution
        self._layers.append(MaskedConv2d(channels_out=self.channels_out, kernel_size=3, input_channels=self.input_channels, mask_type='B',
                                         factorized=self.factorized))
        self._layers.append(tf.keras.layers.Activation("relu"))
        # Upsample channels by two using 1x1 convolution
        self._layers.append(MaskedConv2d(channels_out=self.channels_out * 2, kernel_size=1, input_channels=self.input_channels, mask_type='B',
                                         factorized=self.factorized))

    def call(self, x, **kwargs):
        x_in = x
        for layer in self._layers:
            x = layer(x)
        return x + x_in


class DSModel(tf.keras.Model):

    def __init__(self, H, W, C, n_vals, factorised, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_vals = n_vals
        self.H = H
        self.W = W
        self.N = n_vals
        self.C = C
        self.factorized = factorised

    def build(self, input_shape, **kwargs):
        self._layers = []
        self._layers.append(MaskedConv2d(channels_out=128 * 2, kernel_size=7, input_channels=self.C, mask_type='A',
                                          factorized=self.factorized))
        for i in range(12):
            self._layers.append(ResBlock(channels_out=128, input_channels=self.C, factorized=self.factorized))
        self._layers.append(tf.keras.layers.Activation("relu"))
        self._layers.append(MaskedConv2d(channels_out=128, kernel_size=1, input_channels=self.C, mask_type='B',
                                          factorized=self.factorized))
        self._layers.append(tf.keras.layers.Activation("relu"))
        self._layers.append(MaskedConv2d(channels_out=self.N * self.C, kernel_size=1, input_channels=self.C, mask_type='B',
                                          factorized=self.factorized))

    def call(self, inputs, training=None, mask=None):
        x = tf.cast(inputs, tf.float32)
        for layer in self._layers:
            x = layer(x)
        x_rshp = tf.reshape(x, [-1, self.H, self.W, self.C, self.N])
        return x_rshp

    def loss(self, inputs):
        inp = tf.cast(inputs, tf.int32)
        x_rshp = self(inputs)  # call self
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inp, logits=x_rshp)
        loss = tf.reduce_mean(losses) * np.log2(np.e)
        return loss


class PixelCNNDS:
    def __init__(self, H=28, W=28, C=3, N=4, factorized=False, learning_rate=10e-4):
        self.name = "PixelCNN-DS-DEBUG"
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.factorized = factorized
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.setup_model()

    def __str__(self):
        return "Name: {}\nFactorised: {}\nLearning rate: {}\n".format(self.name,
                                                                    self.factorized, self.learning_rate)

    def setup_model(self):
        self.model = DSModel(self.H, self.W, self.C, self.N, self.factorized)

    @tf.function
    def forward_softmax(self, X):
        x_rshp = self.model(X)
        logits_64 = tf.cast(x_rshp, tf.float64)
        probs = tf.nn.softmax(logits_64)
        return probs

    def train_step(self, X):
        with tf.GradientTape() as tape:
            logprob = self.eval(X)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob.numpy()

    @tf.function
    def eval(self, X):
        return self.model.loss(X)

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


def compare_masks():
    """
    Compare my masking to DS'
    """
    kernel_size = 4
    in_c = 5
    out_c = 3 * 4
    n_c = 3
    mask_letter = "A"
    factorised = False
    mask = get_pixelcnn_mask(kernel_size, in_c, out_c, mask_letter == "A", n_c, factorised)
    display_mask(mask, None)
    mask_ds = get_mask(kernel_size, in_c, out_c, n_c, mask_letter, factorised)
    display_mask(mask_ds, None)


def test_compare_masks(n=1000):
    """
    Test n different random parameters that the masks are equal.
    """
    np.random.seed(1)
    not_similar = 0
    for i in range(n // 2):
        kernel_size = np.random.randint(1, 18, 1)[0]
        in_c = np.random.randint(1, 512, 1)[0]
        out_c = np.random.randint(1, 512, 1)[0]
        n_c = np.random.randint(1, 9, 1)[0]
        factorised = np.round(np.random.random(1))[0]  # bool
        mask_type = np.random.choice(["A", "B"], 1)[0]
        ds = get_pixelcnn_mask(kernel_size, in_c, out_c, mask_type=="A", n_c, factorised)
        me = get_mask(kernel_size, in_c, out_c, n_c, mask_type, factorised)
        if not np.allclose(ds, me):
            print("Kernel size: {}, In: {}, Out: {}, N: {}, mask type: {}, Factorised: {}".format(kernel_size, in_c,
                                                                                                  out_c, n_c,
                                                                                                  mask_type=="A",
                                                                                                  factorised))
            not_similar += 1
    print("\n{} / {} masks matched".format(n - not_similar, n))
    if not_similar == 0:
        print("PASSED!")


def display_mask_reshape(masks, kernel_size, n, c, i):
    """
    Displays the ith input channel's masks in N x C grid
    """
    # in channels as batch dimension
    mask = masks[:, :, i]
    # use tf reshape
    mask = np.array(tf.reshape(mask, (kernel_size, kernel_size, c, n)))
    print(mask.shape)
    mask_flat = np.concatenate(mask.transpose([2, 1, 0, 3]), axis=0).transpose([2, 1, 0])
    print(mask_flat.shape)
    # data is shape (n, h, w, c)
    # plots n rows of images
    disp = np.concatenate(mask_flat, axis=0)
    print(disp.shape)
    plot_image(disp, "figures/1_3", None)


def tf_reshape_masks():
    """
    See how TF reshapes the output channels into channels and values for softmax
    """
    kernel_size = 4
    in_c = 5
    C = 3  # n channels
    N = 4  # n vals
    out_c = C * N
    mask_letter = "A"
    factorised = False
    i = 4
    mask = get_pixelcnn_mask(kernel_size, in_c, out_c, mask_letter == "A", C, factorised)
    print("Mask in shape", mask.shape)
    display_mask_reshape(mask, kernel_size, N, C, i)
    mask_ds = get_mask(kernel_size, in_c, out_c, C, mask_letter, factorised)
    print("Mask DS in shape", mask_ds.shape)
    display_mask_reshape(mask_ds, kernel_size, N, C, i)


### FOR RUN DS AS IS

# Copied from high_dimensional_data.py else circular dependency
def load_data():
    """
    Not a perfect split as should have some validation data.
    Uses test data as validation here since our interest is generative.
    :return: train data, test data as tf.data.Datasets
    """
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = tf.data.Dataset.from_tensor_slices(data["train"])
    test_data = tf.data.Dataset.from_tensor_slices(data["test"])
    return train_data, test_data


def create_dataset(dataset, batch_size):
    dataset = dataset.shuffle(10000)   # Shuffle the data
    dataset = dataset.batch(batch_size)  # Create batches of data
    return dataset


def run_DS(seed=123):
    # seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_data, test_data = load_data()
    nrof_epochs = 2  # TODO: was 5
    batch_size = 64
    factorized = False

    pixel_cnn_model = PixelCNNDS(28, 28, 3, 4, factorized=factorized)

    train_iterator = create_dataset(train_data, batch_size)

    train_loss_list = []
    for epoch in range(1, nrof_epochs + 1):
        for i, batch in enumerate(train_iterator):
            loss_ = pixel_cnn_model.train_step(batch)
            train_loss_list += [loss_]
            if i % 25 == 0:
                print('train epoch: %4d  batch: %4d  loss: %7.3f' % (epoch, i, loss_))

    test_loss_list = []
    loss_ = pixel_cnn_model.eval_dataset(test_data)
    test_loss_list += [loss_]
    print('test epoch: %d  loss: %.3f' % (epoch, np.mean(test_loss_list)))

    np.random.seed(42)
    images = pixel_cnn_model.get_samples(16)
    display_image_grid(images, "logs/1_3/DS", "DS_samples")


if __name__ == "__main__":
    # compare_sampling()
    # compare_masks()
    # tf_reshape_masks()
    # test_compare_masks()
    run_DS()