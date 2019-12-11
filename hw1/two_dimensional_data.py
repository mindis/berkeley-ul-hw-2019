import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from mlp_model import MLP_Model

"""
distribution: 2D array where (i, j) is p(x1 = i, x2 = j)
"""

def load_distribution():
    return np.load("distribution.npy")


def plot_distribution_heatmap(data):
    sns.heatmap(data)
    plt.show()


def sample_distribution(distribution):
    """
    Samples 100,000 points
    Returns train (70%), val (10%), test (20%) samples as (N, 2) where N is % of 100,000 points
    """
    distribution_flat = np.ravel(distribution)
    n_samples = 100000
    sample_flat = tfp.distributions.Categorical(probs=distribution_flat).sample((n_samples,))
    samples = np.array(np.unravel_index(sample_flat, distribution.shape)).T
    return samples[:int(n_samples * 0.7)], samples[int(n_samples * 0.7):int(n_samples * 0.8)], \
           samples[int(n_samples * 0.8):]


def plot_samples(samples):
    samples_cum = np.zeros((200, 200))
    for s in samples:
        samples_cum[s[0], s[1]] += 1
    plot_distribution_heatmap(samples_cum / len(samples))


def get_batch(X, bs):
    batch_size = min(len(X) - 1, bs)
    inds = np.random.choice(np.arange(len(X) - 1), batch_size, replace=False)
    return X[inds]


class TrainingLogger:
    def __init__(self):
        self._i = []
        self._train = []
        self._val = []

    def add(self, i, train, val):
        print("{:>10}: Train: {:>10.3f}, Val: {:>10.3f}".format(i, train, val))
        self._i.append(i)
        self._train.append(train)
        self._val.append(val)

    def plot(self, test_set_logprob):
        plt.plot(self._i, self._train, label="Train")
        plt.plot(self._i, self._val, label="Validation")
        plt.axhline(y=test_set_logprob, label="Test set", linestyle="--", color="g")
        plt.legend()
        plt.title("Train and Validation Log Probs during learning")
        plt.xlabel("# epochs")
        plt.ylabel("Log prob (bits)")
        plt.savefig("figures/train.svg")
        plt.show()


def run_model(model, X_train, X_val):
    training_logger = TrainingLogger()
    for i in range(1001):
        logprob = model.train_step(get_batch(X_train, 10000))
        if i % 100 == 0:
            val_logprob = model.eval(X_val)
            training_logger.add(i, logprob, val_logprob)


if __name__ == "__main__":
    # get data
    distribution = load_distribution()
    # plot_distribution_heatmap(distribution)
    X_train, X_val, X_test = sample_distribution(distribution)
    # plot_samples(X_train)

    # mlp model
    run_model(MLP_Model(), X_train, X_val)

