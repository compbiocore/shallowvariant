import tensorflow as tf
import numpy as np
import math

def make_model(cnn_size=100, kernel_size=3, learning_rate=0.01):
    num_classes = 3

    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          cnn_size, kernel_size, strides=(1, 1), padding="same", use_bias=True
      ),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

def train(model, dataset):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,TBD, TBD)
    :param train_labels: train labels (all labels for training) of shape (num_labels,3)
    :return: None
    """

    shuffled = dataset.shuffle(buffer_size=100).batch(10)

    model.fit(shuffled, epochs=10)


def test(model, dataset):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param dataset: test data (all inputs for testing) of shape (num_inputs,)
    :returns: loss of the test set

    Note: perplexity is exp(total_loss/number of predictions)
    """

    batched = dataset.batch(10)

    return model.evaluate(batched)
