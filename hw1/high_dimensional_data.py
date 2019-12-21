import numpy as np
from pixelCNN import PixelCNN, display_image_grid, plot_image
from utils import get_batch, TrainingLogger


def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    np.random.shuffle(train_data)
    return train_data[int(len(train_data) * pct_val):], train_data[:int(len(train_data) * pct_val)], data["test"]


def set_seed(seed=100):
    np.random.seed(seed)


def train_model(X_train, X_val, model, training_logger, n_iters=2500, bs=128, log_every=100):
    for i in range(n_iters+1):
        batch = get_batch(X_train, bs)
        logprob = model.train_step(batch)
        if i % log_every == 0:
            # TODO: use full val data
            # get validation performance and add to logs
            val_logprob = model.eval_batch(X_val[:bs*10])
            training_logger.add(i, logprob, val_logprob)
            # draw some samples for visualising training performance
            sample_model(model, 4)


def eval_model(model, X_test, training_logger, bs=128):
    sample_model(model, 100)
    test_logprob = model.eval_batch(X_test[:bs*10])
    training_logger.plot(float(test_logprob))


def sample_model(model, n):
    samples = model.get_samples(n)
    display_image_grid(samples, "Samples from PixelCNN")


def pixel_cnn_main():
    model = PixelCNN()
    X_train, X_val, X_test = load_data()
    train_and_eval_main(X_test, X_train, X_val, model)


def train_and_eval_main(X_test, X_train, X_val, model):
    training_logger = TrainingLogger(model.name, "1_3")
    train_model(X_train, X_val, model, training_logger)
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


def pixel_cnn_debug(one_pattern=False):
    """
    one_pattern: if true repeats a single example to debug overfitting to one sample
    otherwise a sample of perturbed checkerboards are used.
    """
    model = PixelCNN(H=4, W=4)
    X_train, X_val, X_test = debug_data()
    if one_pattern:
        X_train = np.repeat(X_train[0][None], 1000, axis=0)
        train_and_eval_main(X_train, X_train, X_train, model)
    else:
        train_and_eval_main(X_test, X_train, X_val, model)


def plot_data():
    X_train, X_val, X_test = load_data()
    display_image_grid(X_train[:9], "Training-Data")


if __name__ == "__main__":
    set_seed()

    # plot_data()

    # pixel_cnn_debug()
    #TODO: try incr. lr
    pixel_cnn_main()
