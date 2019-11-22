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
                  loss=tf.keras.losses.CategoricalCrossentropy())

    return model

def loss(prbs, labels):
    """
    Calculates average cross entropy sequence to sequence loss of the prediction

    :param prbs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
    :param labels: matrix of shape (batch_size, window_size) containing the labels
    :return: the loss of the model as a tensor of size 1
    """

    # We recommend using tf.keras.losses.sparse_categorical_crossentropy
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

    return tf.reduce_sum(
        tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
    )

def accuracy(prbs, labels):
    """
    Calculates the model's prediction accuracy by comparing
    probabilities to correct labels – no need to modify this.
    :param prbs: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

    :return: the accuracy of the model as a Tensor
    """
    correct_predictions = tf.equal(tf.argmax(prbs, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


def train(model, dataset):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,TBD, TBD)
    :param train_labels: train labels (all labels for training) of shape (num_labels,3)
    :return: None
    """

    shuffled = dataset.shuffle(buffer_size=100).batch(10)

    model.fit(shuffled, epochs=2)


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
