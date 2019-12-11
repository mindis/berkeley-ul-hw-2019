import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns


def sample_data():
    """
    Generates 10,000 data samples
    Returns train data (60%), train data (20%), test data samples (20%)
    """
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    return data[:int(count * .6)], data[int(count * .6):int(count * .8)], data[int(count*.8):]


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


def plot_data(X):
    sns.countplot(X,  color="cyan")
    plt.title("Data")
    plt.xscale("linear", base=10)
    plt.savefig("figures/data.svg")
    plt.show()


def main():
    """
    Runs the program
    """
    # setup
    X_train, X_val, X_test = sample_data()
    model = MLE()
    train_log = TrainingLogger()
    # train
    train(X_train, X_val, model, train_log)
    # eval
    test_preds = model.forward(X_test)
    test_set_logprob = float(model.sum_logprob(test_preds))
    print("Test set logprob: {:.3f}".format(test_set_logprob))
    # plot
    train_log.plot(test_set_logprob)
    plot_model(model)
    plot_data(X_train)


def train(X_train, X_val, model, train_log):
    for i in range(201):
        logprob = model.train_step(X_train)
        if i % 100 == 0:
            val_logprob = model.eval(X_val)
            train_log.add(i, logprob, val_logprob)


def plot_model(model):
    all_probs, samples = model.get_probs_and_samples()
    plt.bar(model.get_xs(), all_probs)
    plt.title("Model probabilities over x")
    plt.xscale("linear", base=10)
    plt.xlabel("x")
    plt.ylabel("Model probability")
    plt.savefig("figures/prob.svg")
    plt.show()
    sns.countplot(samples, color="cyan")
    plt.xscale("linear", base=10)
    plt.title("Samples")
    plt.savefig("figures/samples.svg")
    plt.show()


def tf_log2(probs):
    nat_log = tf.math.log(probs)
    return nat_log / tf.math.log(tf.constant(2, dtype=nat_log.dtype))


class MLE:
    def __init__(self):
        self.setup_model()

    def setup_model(self):
        self.params = tf.Variable(np.zeros(100))

    @tf.function
    def forward(self, x):
        return tf.exp(tf.gather(self.params, x - 1)) / tf.reduce_sum(tf.exp(self.params))

    @tf.function
    def sum_logprob(self, probs):
        """
        MLE in bits
        """
        return tf.reduce_mean(-tf_log2(probs))

    def train_step(self, X_train):
        opt = tf.optimizers.Adam(learning_rate=0.1)
        with tf.GradientTape() as tape:
            preds = self.forward(X_train)
            logprob = self.sum_logprob(preds)
        grads = tape.gradient(logprob, [self.params])
        opt.apply_gradients(zip(grads, [self.params]))
        return logprob

    def eval(self, X_test):
        preds = self.forward(X_test)
        logprob = self.sum_logprob(preds)
        return logprob

    def get_probs_and_samples(self):
        """
        Returns probabilities for each x 1, ..., 100
        and 1000 samples from the distribution
        """
        probs = self.forward(self.get_xs())
        distb = tfp.distributions.Categorical(probs=probs)
        return probs, np.array(distb.sample((1000,), seed=562))

    def get_xs(self):
        return np.arange(100) + 1


if __name__ == "__main__":
    main()
