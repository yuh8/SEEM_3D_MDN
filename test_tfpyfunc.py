import tensorflow as tf
import numpy as np


def log_huber(x, m):
    if np.abs(x.numpy()) <= m.numpy():
        return tf.convert_to_tensor((np.log(x)**2).astype(np.float32))
    else:
        return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))


if __name__ == '__main__':
    x = tf.constant(2.0)
    m = tf.constant(4.0)
    breakpoint()

    with tf.GradientTape() as t:
        t.watch([x, m])
        y = tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32)

    dy_dx = t.gradient(y, x)
    assert dy_dx.numpy() == 2.0
