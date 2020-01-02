if __name__ == "__main__":
    import numpy as np
    inds = np.arange(15)
    inds = np.reshape(inds, (-1, 3))
    # print(np.triu(inds))
    n_channels = 3
    i, j = np.triu_indices(n_channels)
    print(inds[i::n_channels, j::n_channels])