import tensorflow as tf
import numpy as np
import math


class Model(tf.keras.Model):
    def __init__(self, cnn_size=100, learning_rate=0.01):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param cnn_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size

        self.num_classes = 3
        self.batch_size = 50

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.cnn = tf.keras.layers.Conv2D(cnn_size, kernel_size, strides=(1,1), padding="same", use_bias=True)
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    def call(self, inputs, initial_state):
        """
        :param inputs: "image" of size TBDxTBD
        :return: the batch element probabilities as a tensor
        """
        out_cnn = self.cnn(inputs)
        out_dense = self.dense(out_cnn)

        return out_dense


    def loss(self, prbs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param prbs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

    def accuracy(self, prbs, labels):
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


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,TBD, TBD)
    :param train_labels: train labels (all labels for training) of shape (num_labels,3)
    :return: None
    """
    n = len(train_labels)

    indices = range(n)
    indices = tf.random.shuffle(indices)

    t_inputs = tf.gather(train_inputs, indices)
    t_labels = tf.gather(train_labels, indices)

    for start_ind in range(0, n, model.batch_size):
        inputs = t_inputs[start_ind : start_ind + model.batch_size]
        labels = t_labels[start_ind : start_ind + model.batch_size]

        with tf.GradientTape() as tape:
            prbs = model.call(inputs) # this calls the call function conveniently
            loss = model.loss(prbs, labels)

            if start_ind % (model.batch_size * 100) == 0:
                print("Loss on training set after {} training steps: {}".format(start_ind, loss))

        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)
    """

    n =len(test_labels)

    running_loss = 0
	running_accuracy = 0.0
    for start_ind in range(0, n, model.batch_size):
        inputs = t_inputs[start_ind : start_ind + model.batch_size]
        labels = t_labels[start_ind : start_ind + model.batch_size]

        prbs = model.call(inputs) # this calls the call function conveniently

        loss = model.loss(prbs, labels)
		accuracy = model.accuracy(prbs, labels)

        running_loss += loss
		running_accuracy += accuracy

        if start_ind % (model.batch_size * 100) == 0:
            print("Loss on test set after {} training steps: {}".format(start_ind, loss))

    return running_loss, running_accuracy
