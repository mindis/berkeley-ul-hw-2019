import numpy as np
import matplotlib.pyplot as plt


def load_data(pct_val=0.15):
    data = np.load("mnist-hw1.pkl", allow_pickle=True)
    train_data = data["train"]
    return train_data[:int(len(train_data) * pct_val)], train_data[int(len(train_data) * pct_val):], data["test"]


def set_seed(seed=100):
    np.random.seed(seed)


def plot_image(image, title, n_vals=3):
    # We use values [0, ..., 3] so we rescale colours for plotting
    plt.imshow((image * 255./n_vals).astype(np.uint8))
    plt.title(title)
    plt.savefig("figures/1_3/{}".format(title))
    plt.show()


def display_image_grid(data, title, n_display=3):
    """
    data is shape (bs, h, w, c)
    plots n x n grid of images
    """
    h = data.shape[1]
    w = data.shape[2]
    disp = np.zeros((h * n_display, w * n_display, data.shape[3]))
    for i in range(n_display):
        for j in range(n_display):
            disp[i * h: (i+1) * h, j * h: (j+1) * h] = data[n_display * i + j]
    plot_image(disp, "{}-grid".format(title))


if __name__ == "__main__":
    set_seed()

    # get data
    X_train, X_val, X_test = load_data()
    # display_image_grid(X_train, "Training-Data")
