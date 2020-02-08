import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def gather_nd(x, inds, name="gather_nd"):
    """
    x (bs, N)
    inds (bs,)
    For ith row of x, gathers the inds[i] element.
    """
    return tf.gather_nd(x, tf.expand_dims(inds, axis=-1), batch_dims=tf.rank(inds))


def tf_log2(probs):
    nat_log = tf.math.log(probs)
    return tf_log_to_base_n(nat_log, 2)


def tf_log_to_base_n(nat_log, n):
    """
    natlog = tf.log natural log
    n: base to change to, int
    """
    return nat_log / tf.math.log(tf.constant(n, dtype=nat_log.dtype))

class BatchData:
    def __init__(self, X, bs):
        self.X = X  # data
        self.N = len(self.X)  # number of data rows
        self.bs = min(self.N, bs)  # batch size
        self.inds = np.arange(self.N)  # indices into data
        np.random.shuffle(self.inds)  # random ordering
        self.i = 0  # pointer to current index into data

    def get_batch(self):
        # take next batch of indices with wrap around
        inds = np.take(self.inds, range(self.i, self.i+self.bs), mode="wrap")
        # update pointer
        self.i = (self.i + self.bs) % self.N
        return self.X[inds]


def get_batch(X, bs):
    pass


class TrainingLogger:
    def __init__(self, model_name, q):
        """
        :param model_name: model name for logs and plot titles
        :param q: which question for which directory to store plots in
        """
        self._i = []
        self._train = []
        self._val = []
        self.model_name = model_name
        timestamp = time.strftime("%Y%m%d-%H%M")
        self.log_dir = "logs/{}/{}-{}".format(q, model_name, timestamp)
        os.makedirs(self.log_dir)
        print("Logging to {}".format(self.log_dir))

    def add(self, i, train, val):
        """
        i - iteration
        train, val - set log probabilities in bits per dimension
        """
        print("{} {:>8}:\t Train: {:<6.3f} Val: {:<6.3f}".format(time.strftime("%d %b %Y %H:%M:%S"), i, train, val))
        self._i.append(i)
        self._train.append(train)
        self._val.append(val)

    def plot(self, test_set_logprob, clip_loss_plot=None):
        """
        Give test set sum of negative log likelihoods divided by number of dimensions
        for log probability in bits per dimension
        :param clip_loss_plot: max y value for plot (optional)
        """
        df = pd.DataFrame({"Train": self._train, "Validation": self._val})
        # store logs to file
        log_f = "{}/logs.txt".format(self.log_dir)
        with open(log_f, "w") as f:
            df.to_string(f, index=False)
        # plot logs
        plt.clf()
        df.plot()
        plt.axhline(y=test_set_logprob, label="Test set", linestyle="--", color="g")
        plt.legend()
        plt.title("Train and Validation Log Probs during learning")
        plt.xlabel("# iterations")
        plt.ylabel("Log prob (bits per dimension)")
        if clip_loss_plot is not None:
            plt.ylim(top=clip_loss_plot)
        plt.savefig("{}/{}-train.svg".format(self.log_dir, self.model_name))
        plt.draw()
        plt.pause(0.001)
