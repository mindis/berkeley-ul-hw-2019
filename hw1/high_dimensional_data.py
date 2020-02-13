import numpy as np
import tensorflow as tf
from pixelCNN import PixelCNN, display_image_grid, plot_image
from pixelCNNMADE import PixelCNNMADE
from pixelCNN_DS import PixelCNNDS
from utils import TrainingLogger, BatchData
import argparse


def load_data():
    """
    Not a perfect split as should have some validation data.
    Uses test data as validation here since our interest is generative.
    :return: train data, test data as tf.data.Datasets
    """
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = tf.data.Dataset.from_tensor_slices(data["train"])
    test_data = tf.data.Dataset.from_tensor_slices(data["test"])
    return train_data, test_data


def plot_data():
    X_train, X_test = load_data()
    display_image_grid(X_train[:9], "figures/1_3", "Training-Data")


def set_seed(seed=100):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(X_train, X_test, model, training_logger, n_epochs=3, bs=64, log_every=50):
    """
    Run training loop.
    Note sampling and validation take a while so we do them periodically.
    :param X_train: train data
    :param X_test: test /val data to check loss unseen during training
    :param model: model to train
    :param training_logger: logger to update with train and val scores
    :param n_epochs: number of epochs to complete
    :param bs: batch size
    :param log_every: # batches to log at multiples of
    """
    train_iter = X_train.shuffle(bs * 2).batch(bs)
    i = 0
    for epoch in range(n_epochs):
        for batch in train_iter:
            i += 1
            logprob = model.train_step(batch)
            training_logger.add(i, logprob)
            # log at start then every log_every steps
            if i % log_every == 0 or i == 1:
                # get validation performance and add to logs
                val_logprob = model.eval_dataset(X_test)
                training_logger.add_val(i, val_logprob)
        # draw some samples for visualising training performance each epoch
        print("Finished epoch {}, sampling.".format(epoch))
        sample_and_display(model, 4, training_logger.log_dir, label=" " + str(i))
    # add final val / test logprob if didn't before
    print("Finished training.")
    if i % log_every != 0:
        print("Final val metrics:")
        val_logprob = model.eval_dataset(X_test)
        training_logger.add_val(i, val_logprob)


def eval_model(model, training_logger):
    print("Plotting and sampling.")
    training_logger.plot(ymax=2.5)
    # this can take a while, less samples is quicker, ideally 100
    sample_and_display(model, 16, training_logger.log_dir, label=" final")


def sample_and_display(model, n, dir_path, label=""):
    samples = model.get_samples(n)
    display_image_grid(samples, dir_path, "Samples from PixelCNN" + label)


def pixel_cnn_main(model):
    """
    Run pixel CNN: Loads data, trains model and evaluates final samples and test set
    """
    X_train, X_test = load_data()
    train_and_eval_main(X_train, X_test, model, "main")


def train_and_eval_main(X_train, X_test, model, exp_name, **kwargs):
    training_logger = TrainingLogger(str(model.name) + "-" + str(exp_name), "1_3")
    training_logger.log_config(model)
    train_model(X_train, X_test, model, training_logger, **kwargs)
    eval_model(model, training_logger)


def debug_data(one_pattern, n_samples=1000, pct_test=0.2):
    """
    :param one_pattern: if true then data is all the same example, o/w noise is added
    to make a dataset
    Provides a simple dataset that should be easy to learn for debugging model.
    3 channels each 0, 1, 2 or 3, like real data but size 4 x 4 images
    checkerboard with low as 0 or 1 and high as 2 or 3.
    and some randomly perturbed squares
    """
    pct_train = 1.0 - pct_test
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
    if one_pattern:
        train_data = tf.data.Dataset.from_tensor_slices([data[0]] * int(n_samples * pct_train))
        test_data = tf.data.Dataset.from_tensor_slices([data[0]] * int(n_samples * pct_test))
    else:
        train_data = tf.data.Dataset.from_tensor_slices(data[:int(n_samples * pct_train)])
        test_data = tf.data.Dataset.from_tensor_slices(data[int(n_samples * pct_train):])
    return train_data, test_data


def pixel_cnn_debug(model, one_pattern=True):
    """
    one_pattern: if true repeats a single example to debug overfitting to one sample
    otherwise a sample of perturbed checkerboards are used.
    """
    X_train, X_test = debug_data(one_pattern)
    train_and_eval_main(X_train, X_train, model, "debug", log_every=5, n_epochs=1)


def plot_debug_data():
    X_train, X_test = debug_data(False)
    display_image_grid(X_train[:9], "", None)


def pixel_cnn_few(model, n_in=1, n_out=100):
    """
    Run pixel CNN on subset of full data of size n_in, repeated to make a dataset of
    size n_out.
    Test and val set are same here.
    A debug / sanity check
    """
    X_train, _ = load_data()
    # take n_in and convert to numpy
    data = np.array([x for x in X_train.take(n_in).as_numpy_iterator()])
    if len(data) == 1:
        data = [data]
    tf_data = tf.data.Dataset.from_tensor_slices(data * int(n_out / n_in))
    train_and_eval_main(tf_data, tf_data, model, "few", n_epochs=2, log_every=10)


if __name__ == "__main__":
    set_seed()

    models = {"PixelCNN": PixelCNN, "PixelCNN-MADE": PixelCNNMADE, "PixelCNN-DS": PixelCNNDS}
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

# TODO: code up receptive field visualisations, also do 3 x 3 grid like example masks to show
#   how factorised / full compare in channels input path to output
