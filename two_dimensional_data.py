import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from mlp_model import MLPModel
from MADE import MADE
from utils import get_batch, TrainingLogger

"""
distribution: 2D array where (i, j) is p(x1 = i, x2 = j)
"""

def load_distribution():
    return np.load("distribution.npy")


def plot_distribution_heatmap(data, title):
    sns.heatmap(data)
    plt.title(title)
    plt.savefig("figures/1_2/{}".format(title))
    plt.draw()
    plt.pause(0.001)


def get_dataset(distribution, pct_train=0.65, pct_val=0.15):
    """
    Samples 100,000 points
    Returns train (65%), val (15%), test (20%) samples as (N, 2) where N is % of 100,000 points
    """
    n_samples = 100000
    samples = get_2d_distribution_samples(distribution, n_samples)
    return samples[:int(n_samples * pct_train)], samples[int(n_samples * pct_train):int(n_samples * (pct_train+pct_val))], \
           samples[int(n_samples * (pct_train+pct_val)):]


def get_2d_distribution_samples(distribution, n_samples, seed=100):
    distribution_flat = np.ravel(distribution)
    sample_flat = tfp.distributions.Categorical(probs=distribution_flat).sample((n_samples,), seed=seed)
    samples = np.array(np.unravel_index(sample_flat, distribution.shape)).T
    return samples


def plot_samples(samples, title):
    samples_cum = np.zeros((200, 200))
    for s in samples:
        samples_cum[s[0], s[1]] += 1
    plot_distribution_heatmap(samples_cum / len(samples), title)


def train_model(X_train, X_val, model, training_logger):
    for i in range(1001):
        logprob = model.train_step(get_batch(X_train, 10000))
        if i % 100 == 0:
            val_logprob = model.eval(X_val)
            training_logger.add(i, logprob, val_logprob)


def eval_model(model, X_test, training_logger):
    probs = model.get_probs()
    plot_distribution_heatmap(probs, "{}-distribution".format(model.name))
    test_logprob = model.sum_logprob(model.forward(X_test))
    training_logger.plot(float(test_logprob))


def model_main(model, X_train, X_val, X_test):
    training_logger = TrainingLogger(model.name, "1_2")
    train_model(X_train, X_val, model, training_logger)
    eval_model(model, X_test, training_logger)


def set_seed(seed=100):
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed()

    # get data
    distribution = load_distribution()
    plot_distribution_heatmap(distribution, "True distribution")
    X_train, X_val, X_test = get_dataset(distribution)
    plot_samples(X_train, "Data distribution samples")

    # mlp model
    model_main(MLPModel(), X_train, X_val, X_test)

    # mlp model
    model_main(MADE(), X_train, X_val, X_test)

