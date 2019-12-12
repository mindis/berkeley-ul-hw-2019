import numpy as np
import tensorflow as tf

from utils import tf_log2, gather_nd


def sample_unit_numbers(n_units, n_random_vars, seed=100):
    """
    Sample each unit's number (the max number of inputs) from 1 to D-1 where D is the number of
    random variables in the outputs of the whole model.
    n_units in this layer
    """
    np.random.seed(seed)
    return np.random.randint(1, n_random_vars+1, size=n_units)


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


class MADELayer(tf.keras.layers.Layer):
    def __init__(self, n_units, prev_unit_numbers, n_random_vars,
                 unit_numbers=None, activation="relu"):
        """
        n_units is the number of units in this layer
        For MADE masking:
        prev_unit_numbers is the unit numbers (maximum number of inputs to be connected to) for prev layer
        n_random_vars is the D number of random variables in the models output
        unit_numbers are the numbers for each unit in this layer, if None (default) to random sampling these numbers
        activation to use in this layer default: "relu", can choose None for linear layer
        """
        super().__init__()
        self.units = n_units
        self.prev_unit_numbers = prev_unit_numbers
        if unit_numbers is None:
            unit_numbers = sample_unit_numbers(self.units, n_random_vars)
        self.unit_numbers = unit_numbers
        # settings
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get('glorot_uniform')
        self.bias_initializer = tf.keras.initializers.get('zeros')
        self.kernel_regularizer = tf.keras.regularizers.get(None)
        self.bias_regularizer = tf.keras.regularizers.get(None)
        self.kernel_constraint = tf.keras.constraints.get(None)
        self.bias_constraint = tf.keras.constraints.get(None)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            'bias',
            shape=[self.units, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        self.mask = get_mask_made(self.prev_unit_numbers, self.unit_numbers)

    def call(self, inputs, **kwargs):
        masked_weights = self.mask * self.kernel
        z = tf.matmul(inputs, masked_weights) + self.bias
        return self.activation(z)


class MADEModel(tf.keras.Model):
    def __init__(self, D, N, *args, **kwargs):
        """
        D is the number of variables.
        N is the number of values for this variable, so each of the D variables can take on
        values [0, N-1] for N possible values.
        eg. x1, x2 both [0, 9], D = 2, N = 10
        Because we extend to non-binary, multiple inputs/outputs can be for the same
        output variable (for one-hot/softmax).
        """
        super().__init__(*args, **kwargs)
        self.D = D
        self.N = N

    def build(self, input_shape, **kwargs):
        # initial ordering is 1,1,1 ... N times then 2, 2, 2... N times, until D, D, D... N times
        unit_numbers = np.concatenate([np.ones(self.N) * (i+1) for i in range(self.D)])
        self.layer1 = MADELayer(64, unit_numbers, self.D)
        self.layer2 = MADELayer(64, self.layer1.unit_numbers, self.D)
        # N * D outputs
        self.output_layer = MADELayer(self.N * self.D, self.layer2.unit_numbers, self.D, unit_numbers=unit_numbers,
                                      activation=None)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class MADE:
    def __init__(self):
        self.name = "MADE"
        self.N = 200
        self.D = 2
        self.setup_model()
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)

    def setup_model(self):
        """
        We have D variables (usually 2) which can take N values.
        So we MADE with D * N input and output units but that can only take D MADE unit numbers
        that we then separately softmax to get the D output probabilities
        """
        self.model = MADEModel(self.D, self.N)
        self.model.build((None, self.N * self.D))
        self.trainable_variables = self.model.trainable_variables

    @tf.function
    def forward(self, x):
        # outputs of MADE x1,1 is unconditioned x1,2 | x1,1 up to x2,N | x2,N-1, ..., x1,1
        one_hot = tf.one_hot(x, self.N)
        x_one_hot_flat = tf.reshape(one_hot, (-1, self.N * self.D))
        model_outputs = self.model(x_one_hot_flat)
        x1_outputs, x2_outputs = tf.unstack(tf.reshape(model_outputs, (-1, self.D, self.N)),axis=1)
        # softmax and gather units of interest
        px1s = tf.nn.softmax(x1_outputs)
        px1 = gather_nd(px1s, x[:, 0])

        tf.print(px1s[0])

        px2_given_x1s = tf.nn.softmax(x2_outputs)
        px2_given_x1 = gather_nd(px2_given_x1s, x[:, 1])
        # joints is p(x2|x1)p(x1)
        return px1 * px2_given_x1

    @tf.function
    def sum_logprob(self, probs):
        """
        MLE in bits per dimension
        """
        return tf.reduce_mean(-tf_log2(probs)) / 2.

    def train_step(self, X_train):
        with tf.GradientTape() as tape:
            preds = self.forward(X_train)
            logprob = self.sum_logprob(preds)
        grads = tape.gradient(logprob, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return logprob

    def eval(self, X_test):
        preds = self.forward(X_test)
        logprob = self.sum_logprob(preds)
        return logprob

    def get_probs(self):
        """
        Returns probabilities for each x1, x2, in 0, ..., 199 as a 2D array (i, j) = p(x1=i, x2=j)
        """
        probs_flat = np.squeeze(self.forward(self.get_xs()))
        probs = probs_flat.reshape((200, 200))
        print(np.sum(probs))
        return probs

    def get_xs(self):
        xs = []
        for i in range(self.N):
            x = np.stack([np.ones(self.N, dtype=np.int32) * i, np.arange(self.N, dtype=np.int32)], axis=1)
            xs.append(x)
        xs = np.concatenate(xs, axis=0)
        return xs


def test_masks():
    units = sample_unit_numbers(5, 3)
    print(units)
    print(get_mask_made(np.arange(3) + 1, units))


if __name__ == "__main__":
    test_masks()
