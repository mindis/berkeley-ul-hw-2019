import numpy as np
import tensorflow as tf
from pixelCNN import PixelCNN, display_image_grid, plot_image
from pixelCNNMADE import PixelCNNMADE
from utils import get_batch, TrainingLogger
import argparse


def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    np.random.shuffle(train_data)
    return train_data[int(len(train_data) * pct_val):], train_data[:int(len(train_data) * pct_val)], data["test"]


def plot_data():
    X_train, X_val, X_test = load_data()
    display_image_grid(X_train[:9], "Training-Data")


def set_seed(seed=100):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(X_train, X_val, model, training_logger, n_iters=1500, bs=128, log_every=100, sample_every=500):
    """
    Run training loop.
    Note sampling and validation take a while so we do them periodically.
    :param X_train: train data
    :param X_val: val data
    :param model: model to train
    :param training_logger: logger to update with train and val scores
    :param n_iters: number of iterations
    :param bs: batch size
    :param log_every: iterations to log at multiples of
    :param sample_at: iterations to sample 4 images at multiples of
    """
    for i in range(n_iters+1):
        batch = get_batch(X_train, bs)
        logprob = model.train_step(batch)
        if i % log_every == 0:
            # get validation performance and add to logs
            val_logprob = model.eval_batch(X_val)
            training_logger.add(i, logprob, val_logprob)
        if i % sample_every == 0 and i > 0:
            # draw some samples for visualising training performance
            sample_model(model, 4, " " + str(i))


def eval_model(model, X_test, training_logger, bs=128):
    test_logprob = model.eval_batch(X_test, bs=bs)
    training_logger.plot(float(test_logprob))
    # this can take a while, less samples is quicker, ideally 100
    sample_model(model, 9, " final")


def sample_model(model, n, label=""):
    samples = model.get_samples(n)
    display_image_grid(samples, "Samples from PixelCNN" + label)


def pixel_cnn_main(model):
    """
    Run pixel CNN: Loads data, trains model and evaluates final samples and test set
    """
    X_train, X_val, X_test = load_data()
    train_and_eval_main(X_test, X_train, X_val, model)


def train_and_eval_main(X_test, X_train, X_val, model, **kwargs):
    training_logger = TrainingLogger(model.name, "1_3")
    train_model(X_train, X_val, model, training_logger, **kwargs)
    eval_model(model, X_test, training_logger)


def debug_data(n_samples=10000, pct_val=0.15, pct_test=0.2):
    """
    Provides a simple dataset that should be easy to learn for debugging model.
    3 channels each 0, 1, 2 or 3, like real data but size 4 x 4 images
    checkerboard with low as 0 or 1 and high as 2 or 3.
    and some randomly perturbed squares
    """
    pct_train = 1.0 - pct_val - pct_test
    data = []
    # iterate over all checkerboard patterns
    for low in range(2):
        for high in range(2, 4):
            # 4 x 4 checkerboard
            x = np.array([[low, high], [high, low]])
            sample_base = np.tile(x, (2, 2))
            sample = np.repeat(sample_base[..., None], 3, axis=2)
            samples = np.repeat(sample[None], n_samples // 4, axis=0)
            data.append(samples)
    data = np.vstack(data)
    # add some randomness
    rows = np.arange(len(data))
    inds = np.random.randint(4, size=len(data))
    channel = np.random.randint(3, size=len(data))
    change = np.random.choice([1, -1], size=len(data))
    data[rows, inds, inds, channel] += change
    # clip to valid values
    data = np.clip(data, 0, 3)
    np.random.shuffle(data)
    return data[:int(n_samples * pct_train)], data[int(n_samples * pct_train):int(n_samples * (pct_train+pct_val))], \
           data[int(n_samples * (pct_train+pct_val)):]


def pixel_cnn_debug(model, one_pattern=True):
    """
    one_pattern: if true repeats a single example to debug overfitting to one sample
    otherwise a sample of perturbed checkerboards are used.
    """
    X_train, X_val, X_test = debug_data()
    if one_pattern:
        X_train = np.repeat(X_train[0][None], 1000, axis=0)
        train_and_eval_main(X_train, X_train, X_train, model, log_every=10, n_iters=50)
    else:
        train_and_eval_main(X_test, X_train, X_val, model, log_every=10, n_iters=100)


def plot_debug_data():
    X_train, X_val, X_test = debug_data()
    display_image_grid(X_train[:9], None)


def pixel_cnn_few(model, n=1):
    """
    Run pixel CNN on subset of size n of data, for running on 1 or few images.
    Test and val set are same here.
    A debug / sanity check
    """
    X_train, _, _ = load_data()
    data = X_train[:n]
    if n == 1:
        data = np.repeat(data, 2, axis=0)
    train_and_eval_main(data, data, data, model)


if __name__ == "__main__":
    set_seed()

    models = {"PixelCNN": PixelCNN, "PixelCNN-MADE": PixelCNNMADE}
    tasks = {"debug": pixel_cnn_debug, "few": pixel_cnn_few, "main": pixel_cnn_main}

    parser = argparse.ArgumentParser()
    parser.add_argument("Model", help="Model to train", choices=models.keys())
    parser.add_argument("Dataset", help="Dataset to run model on", choices=tasks.keys())
    args = parser.parse_args()

    model_uninit = models[args.Model]
    task = args.Dataset
    if task == "debug":
        model = model_uninit(H=4, W=4)
    else:
        model = model_uninit()
    tasks[task](model)

# TODO: run main for factorised and full on pixelcnn and pixelcnn-made (on pc / colab)
# TODO: code up receptive field visualisations, also do 3 x 3 grid like example masks to show
#   how factorised / full compare in channels input path to output
