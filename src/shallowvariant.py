from .model import Model, train, test
import argparse
from preprocess import get_data
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--cnnsize", "-c", default=100)
parser.add_argument("--learningrate", "-l", default=0.01)
parser.add_argument("--trainpath", "-t", nargs="?")
parser.add_argument("--testpath", "-v")
parser.add_argument("--save", "-s", action="store_true")
parser.add_argument("--savepath", "-p")
parser.add_argument("--loadpath", "-m")


def run(
    train=False,
    cnn_size=100,
    learning_rate=0.01,
    train_path="",
    test_path="",
    save=False,
    save_path="",
    load_path="",
):

    if train:
        train_data = get_data(train_path)

    test_data = get_data(test_path)

    # Set-up the training step
    if train:
        # initialize model and tensorflow variables
        model = Model(cnn_size=cnn_size, learning_rate=learning_rate)
        train(model, train_inputs, train_labels)

        if save:
            tf.saved_model.save(model, save_path)
    else:
        model = tf.saved_model.load(load_path)

    # Set up the testing steps
    loss, accuracy = test(model, test_inputs, test_labels)

    # Print out perplexity
    print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train:
        assert args.trainpath != ""

        if args.save:
            assert args.savepath != ""
    else:
        assert args.loadpath != ""

    run(
        train=args.train,
        cnn_size=args.cnnsize,
        learning_rate=args.learningrate,
        train_path=args.trainpath,
        test_path=args.testpath,
        save=args.save,
        save_path=args.savepath,
        load_path=args.loadpath,
    )
