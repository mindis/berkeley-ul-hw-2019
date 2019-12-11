import tensorflow as tf


def gather_nd(x, inds, name="gather_nd"):
    """
    For ith row of x, gathers the inds[i] element.
    """
    indices = tf.stack([tf.range(tf.shape(inds)[0], dtype=inds.dtype), inds], axis=1)
    return tf.gather_nd(x, indices, name=name)


def tf_log2(probs):
    nat_log = tf.math.log(probs)
    return nat_log / tf.math.log(tf.constant(2, dtype=nat_log.dtype))