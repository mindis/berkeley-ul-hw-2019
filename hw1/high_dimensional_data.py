import numpy as np
from pixelCNN import PixelCNN, display_image_grid
from utils import get_batch, TrainingLogger


def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    return train_data[:int(len(train_data) * pct_val)], train_data[int(len(train_data) * pct_val):], data["test"]


def set_seed(seed=100):
    np.random.seed(seed)


def train_model(X_train, X_val, model, training_logger):
    n_iters = 1001
    for i in range(n_iters):
        logprob = model.train_step(get_batch(X_train, 1000))
        if i % 100 == 0:
            val_logprob = model.eval(X_val)
            training_logger.add(i, logprob, val_logprob)


def eval_model(model, X_test, training_logger):
    test_logprob = model.sum_logprob(model.forward(X_test))
    training_logger.plot(float(test_logprob))


def model_main(model, X_train, X_val, X_test):
    training_logger = TrainingLogger(model.name)
    train_model(X_train, X_val, model, training_logger)


if __name__ == "__main__":
    set_seed()

    # get data
    X_train, X_val, X_test = load_data()
    # display_image_grid(X_train[:9], "Training-Data")

    pixelcnn = PixelCNN()
    model_main(pixelcnn, X_train, X_val, X_test)
