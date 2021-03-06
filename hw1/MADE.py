import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers import Dense

from utils import gather_nd, tf_log_to_base_n


def sample_unit_numbers(n_units, min_n, D):
    """
    D is number of random vars in the outputs of the whole model
    Sample each unit's number (the max number of inputs) from 1 to D-1
    n_units in this layer
    min_n is the lowest number to use, avoids disconnected units
    """
    # np upperbound excluded
    return np.random.randint(min_n, D, size=n_units)


def ordered_unit_numbers(n_units, min_n, D):
    """
    Ordered unit numbers for an arbitrary number of units ie. 1, 1, 1, (y times) then 2, 2, 2, (y times),
    ..., D-1, D-1, D-1 (n_units % (D-1) (leftovers) times)
    """
    rep = int(np.ceil(n_units / (D - min_n)))
    layer_units = np.repeat(np.arange(min_n, D), rep)[:n_units]
    return layer_units


def ordered_input_unit_numbers(D, N):
    """
    :param D: the number of RVs
    :param N: number of values each RV can take (because we one-hot it so each value of each RV is a single RV)
    For input and output layers as can only have D x N values
    Gives ordered units so we can sample by conditioning sequentially. ie. 1, 1, 1, (N times) then 2, 2, 2, (N times),
    ..., D, D, D (N times)
    This goes to unit number D as input and output go to D inclusive.
    """
    return np.repeat(np.arange(1, D+1), N)


def get_input_unit_numbers(D, N):
    """
    :param D: the number of RVs
    :param N: number of values each RV can take (because we one-hot it so each value of each RV is a single RV)
    :param N_aux: number of auxiliary dimensions of input we have (other than input D * N)
    So we expect inputs of shape (bs, D * N + N_aux)
    where the first D * N are ordered unit numbers repeated N times ie. 1, 1, 1, ..., D, D, D
    then for last N_aux inputs we have 1's because the aux is assumed to be conditioned on so unmasked
    """
    unit_numbers = ordered_input_unit_numbers(D, N)
    return unit_numbers


def get_output_unit_numbers(D, N):
    unit_numbers = ordered_input_unit_numbers(D, N)
    return unit_numbers


def get_aux_unit_numbers(D, N, N_aux, is_output):
    aux_unit_numbers = np.ones(N_aux)
    # aux_unit_numbers = ordered_input_unit_numbers(D, N)
    if is_output:
        return aux_unit_numbers - 1  # bc output is strict ineq
    return aux_unit_numbers


def get_mask_made(prev_unit_numbers, unit_numbers, is_output):
    """
    is_output: true if an output layer since uses different masking.
    Gets the matrix to multiply with the weights to mask connections for MADE.
    Unit numbers are the max number of inputs that the unit can be connected to (this means inputs not units in the
    previous layer, except in first hidden layer when prev layer is the inputs)
    Masking is different for output layers
    Returns mask (prev_units, units)
    """
    unit_masks = []
    if is_output:
        for prev_unit_number in prev_unit_numbers:
            unit_masks.append(unit_numbers > prev_unit_number)
    else:
        for prev_unit_number in prev_unit_numbers:
            unit_masks.append(unit_numbers >= prev_unit_number)
    return np.array(np.stack(unit_masks), dtype=np.float32)


def one_hot_inputs(x, D, N):
    """
    :param x: inputs
    :param D: the number of RVs
    :param N: number of values each RV can take (because we one-hot it so each value of each RV is a single RV)
    Converts inputs to one hot and reshapes for MADE
    :return: processed inputs (-1, N*D)
    """
    x = tf.cast(x, tf.int32)
    one_hot = tf.one_hot(x, N)
    x = tf.reshape(one_hot, (-1, N * D))
    return x


class MADELayer(Dense):
    def __init__(self, n_units, prev_unit_numbers, D, N,
                 unit_numbers=None, activation="relu", is_output=False,
                 N_aux=0, **kwargs):
        """
        n_units is the number of units in this layer
        For MADE masking:
        prev_unit_numbers is the unit numbers (maximum number of inputs to be connected to) for prev layer
        D is number of random variables in the models output
        unit_numbers are the numbers for each unit in this layer, if None (default) to random sampling these numbers
        """
        super().__init__(n_units, activation=activation, **kwargs)
        if N_aux > 0:  # add aux to prev unit numbers
            aux_unit_numbers = get_aux_unit_numbers(D, N, N_aux, is_output)
            prev_unit_numbers = np.hstack([prev_unit_numbers, aux_unit_numbers])
        if unit_numbers is None:
            unit_numbers = ordered_unit_numbers(n_units, np.min(prev_unit_numbers), D)
        self.prev_unit_numbers = prev_unit_numbers
        self.unit_numbers = unit_numbers
        self.is_output = is_output

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = get_mask_made(self.prev_unit_numbers, self.unit_numbers, self.is_output)

    def call(self, inputs, **kwargs):
        # input shape (bs, n_in + n_aux)
        # mask kernel for internal op, but then return to copy of kernel after for learning
        kernel_copy = self.kernel
        self.kernel = self.kernel * self.mask
        out = super().call(inputs)
        self.kernel = kernel_copy
        return out


class MADEModel(tf.keras.Model):
    def __init__(self, D, N, n_hidden_units, N_aux=0, *args, **kwargs):
        """
        D is the number of variables.
        N is the number of values for this variable, so each of the D variables can take on
        n_hidden_units is number of units in hidden layers
        N_aux is the number of auxiliary input dimensions (default 0)
        values [0, N-1] for N possible values.
        eg. x1, x2 both [0, 9], D = 2, N = 10
        Because we extend to non-binary, multiple inputs/outputs can be for the same
        output variable (for one-hot/softmax).
        """
        super().__init__(*args, **kwargs)
        self.D = D
        self.N = N
        self.n_hidden_units = n_hidden_units
        self.N_aux = N_aux

    def build(self, input_shape, **kwargs):
        # get ordered unit numbers for inputs with aux
        in_unit_numbers = get_input_unit_numbers(self.D, self.N)
        self.layer1 = MADELayer(self.n_hidden_units, in_unit_numbers, self.D, self.N, N_aux=self.N_aux)
        self.layer2 = MADELayer(self.n_hidden_units, self.layer1.unit_numbers, self.D, self.N, N_aux=self.N_aux)
        # N * D outputs
        # Ordered unit numbers for output
        out_unit_numbers = get_output_unit_numbers(self.D, self.N)
        self.output_layer = MADELayer(self.N * self.D, self.layer2.unit_numbers, self.D, self.N, N_aux=self.N_aux,
                                      unit_numbers=out_unit_numbers, activation=None, is_output=True)

    def call(self, inputs, training=None, mask=None):
        if self.N_aux > 0:
            x, aux = inputs
            x = self.layer1(tf.concat([x, aux], axis=-1))
            x = self.layer2(tf.concat([x, aux], axis=-1))
            x = self.output_layer(tf.concat([x, aux], axis=-1))
        else:
            x = inputs
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.output_layer(x)
        x_i_outputs = tf.reshape(x, (-1, self.D, self.N))
        return x_i_outputs


class MADE:
    def __init__(self, name="MADE", N=200, D=2, one_hot=True, n_hidden_units=64, learning_rate=10e-3,
                 grad_clip=1):
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
        self.grad_clip = grad_clip

    def setup_model(self):
        """
        We have D variables (usually 2) which can take N values.
        So we MADE with D * N input and output units but that can only take D MADE unit numbers
        that we then separately softmax to get the D output probabilities
        """
        self.model = MADEModel(self.D, self.N, self.n_hidden_units)

    def forward_logits(self, x):
        """
        Get outputs from model (logits for softmax)
        """
        if self.one_hot:
            x = one_hot_inputs(x, self.D, self.N)
        model_outputs = self.model(x)
        return model_outputs

    @tf.function
    def forward_softmax(self, x):
        """
        Apply softmax over N values to each D variable outputs
        """
        x_i_outputs = self.forward_logits(x)
        # softmax and gather units of interest
        pxis = tf.nn.softmax(x_i_outputs, axis=-1)
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
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
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
    print(get_mask_made(units, hidden_units1, False).T)
    print(hidden_units1)
    print(get_mask_made(hidden_units1, hidden_units2, False).T)
    print(hidden_units1)
    # -1 because the output layer is a strict inequality
    print(get_mask_made(hidden_units2, units - 1, is_output=True).T)


if __name__ == "__main__":
    test_masks()
