import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils import tf_log2, gather_nd


def get_pixelcnn_mask(this_filters_shape, cur_pixel, isTypeA, n_channels=3):
    """
    this_filters_shape define the layer sizes
    isTypeA: bool, true if type A mask, otherwise type B mask used.
    Type A takes context and previous channels (but not its own channel)
    Type B takes context, prev channels and connected to own channel.

    We group the filters so that different filters correspond to different channels.
    first 3 are R, next 3 are G, last 3 are B

    Returns mask of shape (kernel_size, kernel_size, # channels, # filters)
    """
    mask = np.ones(this_filters_shape)
    # mask out all pixels conditioned on
    for i in range(this_filters_shape[0]):
        for j in range(this_filters_shape[1]):
            # deal with centre pixel, then by ordering the other pixels are all 1
            if i == cur_pixel[0] and j == cur_pixel[1]:
                mask_centre(cur_pixel, isTypeA, mask, n_channels)
                break
            # mask all channels for conditioned pixels
            # TODO: should this be other way? due to image y axis flipped
            elif i < cur_pixel[0] or (i == cur_pixel[0] and j < cur_pixel[1]):
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


class MaskedCNN(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 isTypeA,
                 strides=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        """
        Most args see tf.keras.Conv2D.
        isTypeA is for the mask type
        Mostly copied from tf.keras.Conv2D and tf.keras.Conv to adapt to masked Conv2D layer
        """
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        # TODO: we must use same padding?
        self.padding = "SAME"
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.isTypeA = isTypeA

    def build(self, input_shape):
        if input_shape.dims[-1].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[-1])
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        x is the inputs, [image, (cx, cy)] where image is the image / filters from previous layer
        to apply the conv to, and cv, cy are the current image pixel for masking.
        """
        img, cur_pixel = x
        mask = get_pixelcnn_mask(self.kernel_shape, cur_pixel, self.isTypeA)
        masked_kernel = mask * self.kernel
        # TODO: dilation and/or data_format?
        conv_outputs = tf.nn.conv2d(img, masked_kernel, strides=self.strides, padding=self.padding)
        conv_outputs += self.bias
        return self.activation(conv_outputs)


class MaskedResidualBlock(tf.keras.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters

    def build(self, input_shape):
        # 1x1 relu filter, then 3x3 then 1x1
        self.layer1 = MaskedCNN(self.n_filters, 1, False, activation="relu")
        self.layer2 = MaskedCNN(self.n_filters, 3, False, activation="relu")
        self.layer3 = MaskedCNN(self.n_filters, 1, False, activation="relu")

    def call(self, x):
        """
        x is the inputs, [image, (cx, cy)]
        """
        img, _ = x
        # other layers take img and cur pixel location
        x = self.layer1(x)
        x = self.layer2(x)
        res_path = self.layer3(x)
        return img + res_path


class PixelCNNModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_size = 4

    def build(self, input_shape, **kwargs):
        self.layer1 = MaskedCNN(128, 7, True)
        self.res_layers = [MaskedResidualBlock(128)] * 12
        self.conv1x1 = [MaskedCNN(self.output_size, 1, False)] * 2
        self.output_layer = tf.keras.layers.Softmax()
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        for layer in self.res_layers:
            x = layer(x)
        for layer in self.conv1x1:
            x = layer(x)
        x = self.output_layer(x)
        return x


class PixelCNN:
    def __init__(self):
        pass

    def setup_model(self):
        self.model = PixelCNNModel()


def plot_image(image, title, n_vals=3):
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255. / n_vals).astype(np.uint8), cmap="gray")
    if title is not None:
        plt.title(title)
        plt.savefig("figures/1_3/{}".format(title))
    plt.show()


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
    cur_pixel = (3, 2)
    mask = get_pixelcnn_mask(mask_shape, cur_pixel, True)
    # index by prev layer's channels then this layer's channels
    # so rows are prev layer's channels, cols are this layer's
    # concat by prev layer channels into row images, then this channel dim is each image in the row (cols)
    mask_disp = np.concatenate(mask.transpose([3, 1, 0, 2]), axis=0).transpose([2, 1, 0])[..., None]
    print(mask_disp.shape)
    # mask_disp = mask.reshape(mask_shape[:2] + (-1, 1)).transpose([2, 0, 1, 3])
    display_image_rows(mask_disp, "Example Mask A")


def test_maskB():
    mask_shape = (5, 5, 3, 3)
    cur_pixel = (3, 2)
    mask = get_pixelcnn_mask(mask_shape, cur_pixel, False)
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
