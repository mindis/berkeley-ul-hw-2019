import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def gather_nd(x, inds, name="gather_nd"):
    """
    x (bs, N)
    inds (bs,)
    For ith row of x, gathers the inds[i] element.
    """
    indices = tf.stack([tf.range(tf.shape(inds)[0], dtype=inds.dtype), inds], axis=1)
    return tf.gather_nd(x, indices, name=name)


def tf_log2(probs):
    nat_log = tf.math.log(probs)
    return tf_log_to_base_n(nat_log)


def tf_log_to_base_n(nat_log, n):
    """
    natlog = tf.log natural log
    n: base to change to, int
    """
    return nat_log / tf.math.log(tf.constant(n, dtype=nat_log.dtype))


def get_batch(X, bs):
    batch_size = min(len(X) - 1, bs)
    inds = np.random.choice(np.arange(len(X) - 1), batch_size, replace=False)
    return X[inds]


class TrainingLogger:
    def __init__(self, model_name, q):
        self._i = []
        self._train = []
        self._val = []
        self.model_name = model_name
        self.q = q

    def add(self, i, train, val):
        """
        i - iteration
        train, val - set log probabilities in bits per dimension
        """
        print("{} {:>8}:\t Train: {:<6.3f} Val: {:<6.3f}".format(time.strftime("%d %b %Y %H:%M:%S"), i, train, val))
        self._i.append(i)
        self._train.append(train)
        self._val.append(val)

    def plot(self, test_set_logprob):
        """
        Give test set sum of negative log likelihoods divided by number of dimensions
        for log probability in bits per dimension
        """
        plt.plot(self._i, self._train, label="Train")
        plt.plot(self._i, self._val, label="Validation")
        plt.axhline(y=test_set_logprob, label="Test set", linestyle="--", color="g")
        plt.legend()
        plt.title("Train and Validation Log Probs during learning")
        plt.xlabel("# iterations")
        plt.ylabel("Log prob (bits per dimension)")
        plt.savefig("figures/{}/{}-train.svg".format(self.q, self.model_name))
        plt.draw()
        plt.pause(0.001)