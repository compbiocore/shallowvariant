from . import model as m
import argparse
from .preprocess import get_data
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--cnnsize", "-c", default=100)
parser.add_argument("--learningrate", "-l", default=0.01)
parser.add_argument("--inputpath", "-i", required=True)
parser.add_argument("--probspath", "-r", nargs="?")
parser.add_argument("--save", "-s", action="store_true")
parser.add_argument("--savepath", "-p")
parser.add_argument("--loadpath", "-m")


def run(
    train=False,
    cnn_size=100,
    learning_rate=0.01,
    input_path="",
    probs_path="",
    save=False,
    save_path="",
    load_path="",
):

    if train:
        dataset, size = get_data(input_path, probs_path)
        split_n = size // 10
        test_dataset = dataset.take(split_n)
        train_dataset = dataset.skip(split_n)

    else:
        test_dataset, _ = get_data(input_path)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    # Set-up the training step
    if train:
        # initialize model and tensorflow variables
        model = m.make_model(cnn_size=cnn_size, learning_rate=learning_rate)
        m.train(model, train_dataset)

        if save:
            tf.saved_model.save(model, save_path)
    else:
        model = tf.saved_model.load(load_path)

    # Set up the testing steps
    loss, accuracy = m.test(model, test_dataset)

    # Print out perplexity
    print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train:
        assert args.probspath !=""

        if args.save:
            assert args.savepath != ""
    else:
        assert args.loadpath != ""

    run(
        train=args.train,
        cnn_size=args.cnnsize,
        learning_rate=args.learningrate,
        input_path=args.inputpath,
        probs_path=args.probspath,
        save=args.save,
        save_path=args.savepath,
        load_path=args.loadpath,
    )
