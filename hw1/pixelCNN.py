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
                mask_centre(cur_pixel, this_filters_shape, isTypeA, mask, n_channels)
                break
            # mask all channels for conditioned pixels
            # TODO: should this be other way? due to image y axis flipped
            elif i < cur_pixel[0] or (i == cur_pixel[0] and j < cur_pixel[1]):
                mask[i, j, :, :] = 0
    return mask


def mask_centre(cur_pixel, filters_shape, isTypeA, mask, n_channels):
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


# TODO: copy mostly from tf.keras.Conv2D
class MaskedCNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


"""
Type A takes context and previous channels (but not its own channel)
"""
class PixelCNNTypeA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


"""
Type B takes context, prev channels and connected to own channel.
"""
class PixelCNNTypeB(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        super().build(input_shape)

    def call(self, x):
        pass


class PixelCNN:
    def __init__(self):
        pass

    def setup_model(self):
        pass


def plot_image(image, title, n_vals=3):
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255./n_vals).astype(np.uint8), cmap="gray")
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
            disp[i * h: (i+1) * h, j * w: (j+1) * w] = data[data_ind]
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