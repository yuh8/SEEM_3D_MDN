import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def one_hot(target):
    oh = np.zeros((target.size, 10))
    oh[np.arange(target.size), target] = 1
    return oh


def categorical_crossentropy(target, output, from_logits=True):
    target = one_hot(target)
    if from_logits:
        output = softmax(output)
    else:
        output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.sum(target * -np.log(output), axis=-1, keepdims=False)


def loss_func(y_true, y_pred):
    loss = tf.py_function(categorical_crossentropy, inp=[y_true, y_pred], Tout=tf.int32)
    return loss


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
breakpoint()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer='adam',
              loss=loss_func,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
