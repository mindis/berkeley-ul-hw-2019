import numpy as np
from pixelCNN import PixelCNN, display_image_grid


def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    return train_data[:int(len(train_data) * pct_val)], train_data[int(len(train_data) * pct_val):], data["test"]


def set_seed(seed=100):
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed()

    # get data
    X_train, X_val, X_test = load_data()
    # display_image_grid(X_train[:9], "Training-Data")

    # 1st let's try get the forward model working
    pixelcnn = PixelCNN()
    print(pixelcnn.forward(X_train[:10]))
