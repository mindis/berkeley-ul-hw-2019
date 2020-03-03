import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers import Dense

from utils import gather_nd, tf_log_to_base_n


def sample_unit_numbers(n_units, min_n, n_random_vars, seed=100):
    """
    Sample each unit's number (the max number of inputs) from 1 to D-1 where D is the number of
    random variables in the outputs of the whole model.
    n_units in this layer
    min_n is the lowest number to use, avoids disconnected units
    """
    np.random.seed(seed)
    # np upperbound excluded
    return np.random.randint(min_n, n_random_vars, size=n_units)


def ordered_unit_number(D, N):
    """
    :param D: the number of RVs
    :param N: number of values each RV can take (because we one-hot it so each value of each RV is a single RV)
    Primarily for input and output layers
    Gives ordered units so we can sample by conditioning sequentially. Ie. the first N values will be unit number
    1 then the next N will be 2, ... the last N will be unit number D
    """
    return np.repeat(np.arange(1, D), N)


def get_mask_made(prev_unit_numbers, unit_numbers):
    """
    Gets the matrix to multiply with the weights to mask connections for MADE.
    Unit numbers are the max number of inputs that the unit can be connected to (this means inputs not units in the
    previous layer, except in first hidden layer when prev layer is the inputs)
    Returns mask (prev_units, units)
    """
    unit_masks = []
    for prev_unit_number in prev_unit_numbers:
        unit_masks.append(unit_numbers >= prev_unit_number)
    return np.array(np.stack(unit_masks), dtype=np.float32)


class MADELayer(Dense):
    def __init__(self, n_units, prev_unit_numbers, n_random_vars,
                 unit_numbers=None, activation="relu", **kwargs):
        """
        n_units is the number of units in this layer
        For MADE masking:
        prev_unit_numbers is the unit numbers (maximum number of inputs to be connected to) for prev layer
        n_random_vars is the D number of random variables in the models output
        unit_numbers are the numbers for each unit in this layer, if None (default) to random sampling these numbers
        """
        super().__init__(n_units, activation=activation, **kwargs)
        self.prev_unit_numbers = prev_unit_numbers
        if unit_numbers is None:
            unit_numbers = sample_unit_numbers(n_units, np.min(prev_unit_numbers), n_random_vars)
        self.unit_numbers = unit_numbers

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = get_mask_made(self.prev_unit_numbers, self.unit_numbers)

    def call(self, inputs, **kwargs):
        # mask kernel for internal op, but then return to copy of kernel after for learning
        kernel_copy = self.kernel
        self.kernel = self.kernel * self.mask
        out = super().call(inputs)
        self.kernel = kernel_copy
        return out


class MADEModel(tf.keras.Model):
    def __init__(self, D, N, n_hidden_units, *args, **kwargs):
        """
        D is the number of variables.
        N is the number of values for this variable, so each of the D variables can take on
        n_hidden_units is number of units in hidden layers
        values [0, N-1] for N possible values.
        eg. x1, x2 both [0, 9], D = 2, N = 10
        Because we extend to non-binary, multiple inputs/outputs can be for the same
        output variable (for one-hot/softmax).
        """
        super().__init__(*args, **kwargs)
        self.D = D
        self.N = N
        self.n_hidden_units = n_hidden_units

    def build(self, input_shape, **kwargs):
        # get ordered unit numbers for inputs
        in_unit_numbers = ordered_unit_number(self.D, self.N)
        self.layer1 = MADELayer(self.n_hidden_units, in_unit_numbers, self.D)
        self.layer2 = MADELayer(self.n_hidden_units, self.layer1.unit_numbers, self.D)
        # N * D outputs
        # Ordered unit numbers for output
        # -1 because the output layer is a strict inequality
        out_unit_numbers = ordered_unit_number(self.D, self.N) - 1
        self.output_layer = MADELayer(self.N * self.D, self.layer2.unit_numbers, self.D, unit_numbers=out_unit_numbers,
                                      activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.output_layer(x)
        x_i_outputs = tf.reshape(x, (-1, self.D, self.N))
        return x_i_outputs


class MADE:
    def __init__(self, name="MADE", N=200, D=2, one_hot=True, n_hidden_units=64, learning_rate=10e-3):
        """
        :param name: model name
        :param N: number of values each var can take
        :param D: number of variables
        :param one_hot: if true then one-hot input before pass to model
        """
        self.name = name
        self.N = N
        self.D = D
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.one_hot = one_hot
        self.n_hidden_units = n_hidden_units
        self.setup_model()

    def setup_model(self):
        """
        We have D variables (usually 2) which can take N values.
        So we MADE with D * N input and output units but that can only take D MADE unit numbers
        that we then separately softmax to get the D output probabilities
        """
        self.model = MADEModel(self.D, self.N, self.n_hidden_units)

    @tf.function
    def forward_logits(self, x):
        """
        Get outputs from model (logits for softmax)
        """
        if self.one_hot:
            x = tf.cast(x, tf.int32)
            one_hot = tf.one_hot(x, self.N)
            x = tf.reshape(one_hot, (-1, self.N * self.D))
        model_outputs = self.model(x)
        return model_outputs

    @tf.function
    def forward_softmax(self, x):
        """
        Apply softmax over N values to each D variable outputs
        """
        x_i_outputs = self.forward_logits(x)
        # softmax and gather units of interest
        pxis = tf.nn.softmax(x_i_outputs)
        return pxis

    @tf.function
    def forward_softmax_gather(self, x):
        """
        Apply softmax over N values to each D variable outputs and gather the probs of the values in x
        """
        pxis = self.forward_softmax(x)
        x_int = tf.cast(x, tf.int32)
        pxi = gather_nd(pxis, x_int)
        # joints is product of conditionals p(x2|x1)p(x1)
        return tf.reduce_prod(pxi, axis=-1)

    @tf.function
    def loss(self, logits, labels):
        """
        MLE in bits per dimension
        Uses numerically stable TF with_logits methods
        """
        # return tf.reduce_mean(-tf_log2(probs)) / tf.cast(self.D, tf.float32)
        labels = tf.cast(labels, tf.int32)
        # labels are (bs, D), logits are (bs, D, N), we want to softmax over each D separately
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # get in bits
        neg_logprob_bit = tf_log_to_base_n(loss, 2)
        return neg_logprob_bit

    def train_step(self, X_train):
        with tf.GradientTape() as tape:
            logprob = self.eval(X_train)
        grads = tape.gradient(logprob, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return logprob.numpy()

    def eval(self, X):
        X_float = tf.cast(X, tf.float32)
        logits = self.forward_logits(X_float)
        logprob = self.loss(logits, X)
        return logprob

    def eval_dataset(self, X, bs=128):
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

    def get_probs(self):
        """
        Returns probabilities for each x1, x2, in 0, ..., 199 as a 2D array (i, j) = p(x1=i, x2=j)
        """
        probs_flat = np.squeeze(self.forward_softmax_gather(self.get_xs()))
        probs = probs_flat.reshape((self.N, self.N))
        return probs

    def get_xs(self):
        xs = []
        for i in range(self.N):
            x = np.stack([np.ones(self.N, dtype=np.int32) * i, np.arange(self.N, dtype=np.int32)], axis=1)
            xs.append(x)
        xs = np.concatenate(xs, axis=0)
        return xs


def test_masks():
    # copy of example in paper, note these will be transposed here to match paper
    units = np.array([3, 1, 2])
    hidden_units1 = np.array([2, 1, 2, 2])
    hidden_units2 = np.array([1, 2, 2, 1])
    print(units)
    print(get_mask_made(units, hidden_units1).T)
    print(hidden_units1)
    print(get_mask_made(hidden_units1, hidden_units2).T)
    print(hidden_units1)
    # -1 because the output layer is a strict inequality
    print(get_mask_made(hidden_units2, units - 1).T)


if __name__ == "__main__":
    test_masks()
