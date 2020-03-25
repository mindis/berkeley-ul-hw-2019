import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def mask(size, type_A):
    m = np.zeros((size, size), dtype=np.float32)
    m[:size // 2, :] = 1
    m[size // 2, :size // 2] = 1
    if not type_A:
        m[size // 2, size // 2] = 1
    return m


def conv_masked(inp, name, size, in_channels=128, out_channels=128, type_A=False):
    conv_filter = tf.compat.v1.get_variable(name=name + '_filter',
                                  shape=(size, size, in_channels, out_channels),
                                  trainable=True)
    conv_bias = tf.compat.v1.get_variable(name=name + '_bias',
                                shape=(inp.shape[1], inp.shape[2], out_channels),
                                trainable=True)

    masked_conv_filter = conv_filter * mask(size, type_A)[:, :, np.newaxis, np.newaxis]
    return tf.nn.conv2d(inp, masked_conv_filter, strides=[1, 1, 1, 1], padding='SAME',
                        name=name) + conv_bias


def conv_1x1(inp, name, in_channels, out_channels):
    conv_filter = tf.compat.v1.get_variable(name=name + '_filter',
                                  shape=(1, 1, in_channels, out_channels),
                                  trainable=True)
    conv_bias = tf.compat.v1.get_variable(name=name + '_bias',
                                shape=(inp.shape[1], inp.shape[2], out_channels),
                                trainable=True)
    return tf.nn.conv2d(inp, conv_filter, strides=[1, 1, 1, 1], padding='SAME', name=name) + conv_bias


def res_block(inp, scope, channels=128):
    with tf.compat.v1.variable_scope(scope):
        res = tf.nn.relu(inp)
        res = conv_1x1(res, 'conv1x1_downsample', channels, channels // 2)
        res = tf.nn.relu(res)
        res = conv_masked(res, 'conv3x3', 3, channels // 2, channels // 2)
        res = tf.nn.relu(res)
        res = conv_1x1(res, 'conv1x1_upsample', channels // 2, channels)

        return inp + res


def pixel_cnn(inp, scope, channels, final_channels):
    with tf.compat.v1.variable_scope(scope):
        net = conv_masked(inp, name='start_conv7x7', size=7, in_channels=3,
                          out_channels=channels, type_A=True)

        for i in range(12):
            net = res_block(net, scope='res_block_{}'.format(i+1),
                            channels=channels)

        net = conv_masked(net, 'final_conv3x3', size=3,
                          in_channels=channels,
                          out_channels=channels)
        net = tf.nn.relu(net)
        net = conv_1x1(net, 'final_conv1x1_1',
                       in_channels=channels,
                       out_channels=channels)
        net = tf.nn.relu(net)
        net = conv_1x1(net, 'final_conv1x1_2',
                       in_channels=channels,
                       out_channels=final_channels)
        return net


class MadeInput:
    def __init__(self, scope, inp, depth):
        with tf.compat.v1.variable_scope(scope):
            self.width = inp.shape[1]
            self.height = inp.shape[2]
            self.D = inp.shape[3]
            self.depth = depth

            self.units = tf.reshape(
                tf.one_hot(inp, depth=self.depth, dtype=tf.float32),
                shape=(tf.shape(inp)[0],
                       self.width,
                       self.height,
                       self.depth * self.D))
            self.m = np.arange(self.D)
            self.m = np.repeat(self.m, self.depth, axis=-1)


class MadeHiddenWithAuxiliary:
    def __init__(self, scope, made_prev_layer, auxiliary, unit_count):
        with tf.compat.v1.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.mod(np.arange(unit_count), self.D - 1)

            self.units = tf.concat([made_prev_layer.units, auxiliary], axis=-1)
            ext_input_length = made_prev_layer.m.shape[-1] + auxiliary.shape[-1]

            self.weight_mask = np.ones((unit_count,
                                        ext_input_length),
                                       dtype=np.bool)
            self.weight_mask[:self.m.shape[-1],
            :made_prev_layer.m.shape[-1]] = self.m[:, np.newaxis] >= made_prev_layer.m[np.newaxis, :]
            self.W = tf.compat.v1.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 unit_count,
                                                 ext_input_length))
            self.b = tf.compat.v1.get_variable("b",
                                     shape=(self.width,
                                            self.height,
                                            unit_count))

            self.units = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask, self.units) + self.b
            self.units = tf.nn.relu(self.units)


class MadeHidden:
    def __init__(self, scope, made_prev_layer, unit_count):
        with tf.compat.v1.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.mod(np.arange(unit_count), self.D - 1)

            self.weight_mask = self.m[:, np.newaxis] >= made_prev_layer.m[np.newaxis, :]
            self.W = tf.compat.v1.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 unit_count,
                                                 made_prev_layer.m.shape[-1]))
            self.b = tf.compat.v1.get_variable("b", shape=(self.width,
                                                 self.height,
                                                 unit_count))

            self.units = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask,
                                   made_prev_layer.units) + self.b
            self.units = tf.nn.relu(self.units)


class MadeOutput:
    def __init__(self, scope, made_input_layer, made_prev_layer, auxiliary):
        with tf.compat.v1.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.repeat(np.arange(self.D), self.depth)
            self.weight_mask = self.m[:, np.newaxis] > made_prev_layer.m[np.newaxis, :]
            self.W = tf.compat.v1.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth,
                                                 made_prev_layer.m.shape[-1]))
            self.b = tf.compat.v1.get_variable("b", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth))

            self.direct_mask = np.repeat(np.tril(np.ones(self.D), -1), self.depth).reshape((self.D, -1))
            self.direct_mask = np.repeat(self.direct_mask, self.depth, axis=0)
            self.A = tf.compat.v1.get_variable("A", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth,
                                                 self.D * self.depth))

            self.units = tf.einsum('whij,bwhj->bwhi',
                                   self.W * self.weight_mask,
                                   made_prev_layer.units) + self.b
            self.units += tf.einsum('whij,bwhj->bwhi',
                                    self.A * self.direct_mask,
                                    made_input_layer.units)

            self.unconnected_W = tf.compat.v1.get_variable('unconnected_W',
                                                 shape=(self.width,
                                                        self.height,
                                                        self.depth,
                                                        auxiliary.shape[-1]))
            self.unconnected_b = tf.compat.v1.get_variable('unconnected_b',
                                                 shape=(self.width,
                                                        self.height,
                                                        self.depth))
            self.unconnected_out = tf.einsum('whij,bwhj->bwhi',
                                             self.unconnected_W,
                                             auxiliary) + self.unconnected_b
            self.units = tf.concat([self.unconnected_out,
                                    self.units[:, :, :, self.depth:]],
                                   axis=3)
            self.units = tf.reshape(self.units, shape=(tf.shape(self.units)[0],
                                                       self.width,
                                                       self.height,
                                                       self.D,
                                                       self.depth))


def sample_image(batch_size, sess, softmaxed, inp):
    image = np.random.choice(4, size=(batch_size, 28, 28, 3)).astype(np.uint8)

    for i in range(28):
        for j in range(28):
            for k in range(3):
                prob_output = sess.run(softmaxed,
                                   {inp: image})
                for b in range(batch_size):
                    image[b, i, j, k] = np.random.choice(4, p=prob_output[b, i, j, k])

    return image


def load_data():
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = np.array_split(data["train"], int(len(data["train"]) / 64))
    test_data = np.array_split(data["test"], int(len(data["test"]) / 64))
    return train_data, test_data

def run():
    tf.compat.v1.reset_default_graph()

    batch_size = 128

    train_iter, test_iter = load_data()

    x_ph = tf.compat.v1.placeholder(tf.int32, shape=(None, 28, 28, 3))

    net = tf.multiply(tf.cast(x_ph, tf.float32), 1. / 3.)
    net = tf.identity(net, 'preprocessed_input')
    net = pixel_cnn(net, scope='pixel_cnn', channels=128, final_channels=16)
    net = tf.nn.relu(net)

    made_input = MadeInput('made_input', x_ph, depth=4)
    made_hidden_1 = MadeHiddenWithAuxiliary('made_h1_with_aux', made_input, net, 32)
    made_out = MadeOutput('made_out', made_input, made_hidden_1, net)
    softmaxed = tf.nn.softmax(made_out.units, axis=-1)

    unreduced_loss = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(x_ph, depth=4),
                                                                made_out.units,
                                                                axis=-1) * np.log2(np.e)
    loss = tf.reduce_mean(unreduced_loss)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = opt.compute_gradients(loss)
    for grad, var in grads_and_vars:
        tf.summary.histogram(grad.name.split(':')[0], grad)
        tf.summary.histogram(var.name.split(':')[0], var)

    grads, variables = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads, 5)
    for grad in grads:
        tf.summary.histogram(grad.name.split(':')[0] + '_clipped', grad)

    train_step = opt.apply_gradients(zip(grads, variables))

    val_loss_mean, val_loss_update = tf.compat.v1.metrics.mean(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    train_losses = []
    val_losses = []


    run = 0
    run += 1

    test_every = 10
    sample_every = 10
    n_epochs = 3

    for e in range(n_epochs):
        for i, batch in enumerate(train_iter):
            _, loss_result, step = sess.run([train_step, loss, global_step],
                                            {x_ph: batch})
            print('\rEpoch {}, step {}\ttrain_loss: {}'.format(
                i, step, loss_result), end='')
            train_losses.append(loss_result)
            if i % sample_every:
                plt.figure(figsize=(12, 12))
                img = sample_image(16, sess, softmaxed, x_ph)

                for k in range(4):
                    for l in range(4):
                        plt.subplot(4, 4, 4 * k + l + 1)
                        plt.imshow(img[4 * k + l] * 255 // 3)
                        plt.axis('off')
                        plt.grid(False)
                        plt.title(loss_result)
                plt.show()
            if i % test_every:
                for test_batch in sess.run(test_iter):
                    v = sess.run(val_loss_update,
                             {x_ph: test_batch})
                    print('\nEpoch {}\t val_loss: {}'.format(i, v))
                    val_losses.append(v)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    run()