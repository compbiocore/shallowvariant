import tensorflow as tf
import numpy as np
from preprocess import get_data
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

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    n = len(train_labels)
    n_batch = math.floor(n / model.window_size)

    train_input_tokens = train_inputs[:n_batch * model.window_size]
    train_inputs = np.reshape(train_input_tokens, (-1, model.window_size))

    train_label_tokens = train_labels[:n_batch * model.window_size]
    train_labels = np.reshape(train_label_tokens, (-1, model.window_size))

    indices = range(n_batch)
    indices = tf.random.shuffle(indices)

    t_inputs = tf.gather(train_inputs, indices)
    t_labels = tf.gather(train_labels, indices)

    for start_ind in range(0, n_batch, model.batch_size):
        inputs = t_inputs[start_ind : start_ind + model.batch_size]
        labels = t_labels[start_ind : start_ind + model.batch_size]

        with tf.GradientTape() as tape:
            prbs, _ = model.call(inputs, None) # this calls the call function conveniently
            loss = model.loss(prbs, labels)

            if start_ind % (model.batch_size * 100) == 0:
                perplexity = np.exp(loss)
                print("Perplixty on training set after {} training steps: {} [loss: {}]".format(start_ind, perplexity, loss))

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
    n_batch = math.floor(n / model.window_size)

    test_input_tokens = test_inputs[:n_batch * model.window_size]
    test_inputs = np.reshape(test_input_tokens, (-1, model.window_size))

    test_label_tokens = test_labels[:n_batch * model.window_size]
    test_labels = np.reshape(test_label_tokens, (-1, model.window_size))

    probs, _ = model.call(test_inputs, None)
    loss = model.loss(probs, test_labels)
    return np.exp(loss)

def generate_sentence(word1, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    from datetime import datetime
    t = datetime.now()
    # TO-DO: Pre-process and vectorize the data
    train_path = './data/train.txt'
    test_path  = './data/test.txt'

    train_tokens, test_tokens, vocab_dict = get_data(train_path, test_path)

    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = train_tokens[:-1]
    train_labels = train_tokens[1:]

    test_inputs = test_tokens[:-1]
    test_labels = test_tokens[1:]

    # TODO: initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(model, test_inputs, test_labels)

    # Print out perplexity
    print("Test Perplexity:", perplexity)

    print("Run time:", datetime.now() - t)

    generate_sentence("down", 20, vocab_dict, model)
    generate_sentence("stock", 20, vocab_dict, model)


if __name__ == '__main__':
    main()
