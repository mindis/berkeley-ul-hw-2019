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
    def __init__(self, H=28, W=28, C=3, N=4, factorized=False, learning_rate=10e-3, seed=1234):
        self.name = "PixelCNN-DS-DEBUG"
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.factorized = factorized
        self.learning_rate = learning_rate
        self.optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.reset_default_graph()
        # set seed
        tf.compat.v1.random.set_random_seed(seed)
        self.sess = tf.compat.v1.Session()
        self.setup_model()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def __str__(self):
        return "Name: {}\nFactorised: {}\nLearning rate: {}\n".format(self.name,
                                                                    self.factorized, self.learning_rate)

    def setup_model(self):
        self.x_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H, self.W, self.C))
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

        x_rshp = tf.reshape(x, [-1, self.H, self.W, self.C, self.N])
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

## FOR RUNNING DS CODE AS WAS

# Copied from high_dimensional_data.py else circular dependency
def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    np.random.shuffle(train_data)
    return train_data[int(len(train_data) * pct_val):], train_data[:int(len(train_data) * pct_val)], data["test"]


def pixel_cnn(x, channels_out, factorized):
    input_channels = x.shape.as_list()[3]
    inp = tf.cast(x, tf.int32)
    x = tf.cast(x, tf.float32)
    with tf.compat.v1.variable_scope('input_conv'):
        x = masked_conv2d(x, channels_out=128 * 2, kernel_size=7, input_channels=input_channels, mask_type='A',
                          factorized=factorized)
    for i in range(12):
        with tf.compat.v1.variable_scope('res_block_%d' % i):
            x = res_block(x, channels_out=128, input_channels=input_channels, factorized=factorized)
    with tf.compat.v1.variable_scope('output_conv_1'):
        x = tf.nn.relu(x)
        x = masked_conv2d(x, channels_out=128, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)
    with tf.compat.v1.variable_scope('output_conv_2'):
        x = tf.nn.relu(x)
        x = masked_conv2d(x, channels_out=channels_out, kernel_size=1, input_channels=input_channels, mask_type='B',
                          factorized=factorized)

    x_rshp = tf.reshape(x, [-1, 28, 28, 3, 4])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inp, logits=x_rshp)
    loss = tf.reduce_mean(losses) * np.log2(np.e)
    probs = tf.nn.softmax(x_rshp)
    return loss, probs, x


def create_dataset(x, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.repeat()   # Repeat the dataset indefinitely
    dataset = dataset.shuffle(10000)   # Shuffle the data
    dataset = dataset.batch(batch_size)  # Create batches of data
    dataset = dataset.prefetch(batch_size)  # Prefetch data for faster consumption
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)  # Create an iterator over the dataset
    return iterator


def sample(sess, eval_loss, eval_probs, eval_input_ph, nrof_images, noise=None):
    img = np.zeros((nrof_images, 28, 28, 3), dtype=np.uint8)
    img[:, 0, 0, 0] = np.random.choice(4, size=(nrof_images,))
    for j in range(28):
        for k in range(28):
            for l in range(3):
                loss, p = sess.run([eval_loss, eval_probs], feed_dict={eval_input_ph:img})
                for i in range(nrof_images):
                    img[i, j, k, l] = np.random.choice(4, p=p[i, j, k, l])
    return img



def run_DS():
    train_data, _, test_data  = load_data(pct_val=0)
    nrof_epochs = 3  # TODO: was 5
    batch_size = 128
    factorized = True

    with tf.Graph().as_default():

        train_iterator = create_dataset(train_data, batch_size)
        test_iterator = create_dataset(test_data, batch_size)
        eval_input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 3))

        with tf.compat.v1.variable_scope('model', reuse=False):
            train_loss, _, _ = pixel_cnn(train_iterator.get_next(), channels_out=3 * 4, factorized=factorized)
        with tf.compat.v1.variable_scope('model', reuse=True):
            test_loss, _, _ = pixel_cnn(test_iterator.get_next(), channels_out=3 * 4, factorized=factorized)
            eval_loss, eval_probs, _ = pixel_cnn(eval_input_ph, channels_out=3 * 4, factorized=factorized)

        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), 1.0)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)

        nrof_train_batches = train_data.shape[0] // batch_size

        train_loss_list = []
        for epoch in range(1, nrof_epochs + 1):
            for i in range(nrof_train_batches):
                _, loss_ = sess.run([train_op, train_loss])
                train_loss_list += [loss_]
                if i % 25 == 0:
                    print('train epoch: %4d  batch: %4d  loss: %7.3f' % (epoch, i, loss_))

        test_loss_list = []
        for i in range(test_data.shape[0] // batch_size):
            loss_ = sess.run([test_loss])
            test_loss_list += [loss_]
        print('test epoch: %d  loss: %.3f' % (epoch, np.mean(test_loss_list)))

        np.random.seed(42)
        images = sample(sess, eval_loss, eval_probs, eval_input_ph, 16)
        display_image_grid(images, "logs/1_3/DS", "DS_samples")


if __name__ == "__main__":
    # compare_sampling()
    # compare_masks()
    # tf_reshape_masks()
    # test_compare_masks()
    run_DS()