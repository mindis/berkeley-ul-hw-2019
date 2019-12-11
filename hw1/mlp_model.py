import numpy as np
import tensorflow as tf

from utils import tf_log2, gather_nd


class FC_Model(tf.keras.Model):
    def __init__(self, N, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(128))
        self.model.add(tf.keras.layers.Dense(128))
        # output layer
        self.model.add(tf.keras.layers.Dense(N))

    def call(self, inputs):
        print("ins", inputs)
        out = self.model(inputs)
        print("outs", out)
        return out


class MLP_Model:
    def __init__(self):
        self.N = 200
        self.setup_model()

    def setup_model(self):
        self.px1_theta = tf.Variable(np.zeros(100), dtype=tf.float32)
        self.x2 = FC_Model(self.N)
        self.x2.build((None, self.N))
        self.trainable_variables = [self.px1_theta] + self.x2.trainable_variables

    @tf.function
    def forward(self, x):
        p_x1 = tf.exp(tf.gather(self.px1_theta, x[:, 0])) / tf.reduce_sum(tf.exp(self.px1_theta))
        x1_one_hot = tf.one_hot(x[:, 0], depth=self.N, dtype=tf.int32)
        p_x2_given_x1_dist = self.x2(x1_one_hot)
        p_x2_given_x1 = p_x2_given_x1_dist[tf.range(tf.shape(x)[0]), x[1]]
        return p_x2_given_x1 * p_x1

    @tf.function
    def sum_logprob(self, probs):
        """
        MLE in bits
        """
        return tf.reduce_mean(-tf_log2(probs))

    def train_step(self, X_train):
        opt = tf.optimizers.Adam(learning_rate=0.01)
        with tf.GradientTape() as tape:
            preds = self.forward(X_train)
            logprob = self.sum_logprob(preds)
        grads = tape.gradient(logprob, [self.trainable_variables])
        opt.apply_gradients(zip(grads, [self.trainable_variables]))
        return logprob

    def eval(self, X_test):
        preds = self.forward(X_test)
        logprob = self.sum_logprob(preds)
        return logprob

