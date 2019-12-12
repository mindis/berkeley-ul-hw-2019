import numpy as np
import tensorflow as tf

from utils import tf_log2, gather_nd


class FCModel(tf.keras.Model):
    def __init__(self, N, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)))
        # output layer
        self.model.add(tf.keras.layers.Dense(N))
        self.model.add(tf.keras.layers.Softmax())

    def call(self, inputs):
        out = self.model(inputs)
        return out


class MLPModel:
    def __init__(self):
        self.name = "MLP"
        self.N = 200
        self.setup_model()
        self.optimizer = tf.optimizers.Adam()

    def setup_model(self):
        self.px1_theta = tf.Variable(np.zeros(self.N), dtype=tf.float32)
        self.x2 = FCModel(self.N)
        # self.x2.build((None, 1))
        self.x2.build((None, self.N))
        self.trainable_variables = [self.px1_theta] + self.x2.trainable_variables

    @tf.function
    def forward(self, x):
        p_x1 = tf.exp(tf.gather(self.px1_theta, x[:, 0])) / tf.reduce_sum(tf.exp(self.px1_theta))
        x1_one_hot = tf.one_hot(x[:, 0], depth=self.N, dtype=tf.int32)
        # p_x2_given_x1_distrbn = self.x2(x[:, 0][:, None])
        p_x2_given_x1_distrbn = self.x2(x1_one_hot)
        p_x2_given_x1 = gather_nd(p_x2_given_x1_distrbn, x[:, 1])
        return p_x2_given_x1 * p_x1

    @tf.function
    def sum_logprob(self, probs):
        """
        MLE in bits per dimension
        """
        return tf.reduce_mean(-tf_log2(probs)) / 2.

    def train_step(self, X_train):
        with tf.GradientTape() as tape:
            preds = self.forward(X_train)
            logprob = self.sum_logprob(preds)
        grads = tape.gradient(logprob, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return logprob

    def eval(self, X_test):
        preds = self.forward(X_test)
        logprob = self.sum_logprob(preds)
        return logprob

    def get_probs(self):
        """
        Returns probabilities for each x1, x2, in 0, ..., 199 as a 2D array (i, j) = p(x1=i, x2=j)
        """
        probs_flat = np.squeeze(self.forward(self.get_xs()))
        probs = probs_flat.reshape((200, 200))
        return probs

    def get_xs(self):
        xs = []
        for i in range(self.N):
            x = np.stack([np.ones(self.N, dtype=np.int32) * i, np.arange(self.N, dtype=np.int32)], axis=1)
            xs.append(x)
        xs = np.concatenate(xs, axis=0)
        return xs

