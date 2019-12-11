from . import model as m
import argparse
from .preprocess import get_data_train, get_data_eval
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--cnnsize", "-c", default=100)
parser.add_argument("--learningrate", "-r", default=0.01)
parser.add_argument("--inputpath", "-i", required=True)
parser.add_argument("--probspath", "-p", nargs="?")
parser.add_argument("--labelpath", "-l", nargs="?")
parser.add_argument("--savepath", "-s", nargs="?")
parser.add_argument("--loadpath", "-m", nargs="?")


def run(
    train=False,
    cnn_size=100,
    learning_rate=0.01,
    input_path="",
    probs_path="",
    label_path="",
    save=False,
    save_path="",
    load_path="",
):

    if train:
        dataset = get_data_train(input_path, probs_path)
        # split_n = size // 10
        # test_dataset = dataset.take(split_n)
        # train_dataset = dataset.skip(split_n)

    else:
        dataset = get_data_eval(input_path, label_path)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    # Set-up the training step
    if train:
        # initialize model and tensorflow variables
        model = m.make_model(cnn_size=cnn_size, learning_rate=learning_rate)
        m.train(model, dataset)

        if save:
            model.save(save_path)
    else:
        model = tf.keras.models.load_model(load_path)

    # Set up the testing steps
    if not train:
        loss, accuracy = m.test(model, dataset)

        # Print out perplexity
        print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.savepath != "":
        save = True

    run(
        train=args.train,
        cnn_size=args.cnnsize,
        learning_rate=args.learningrate,
        input_path=args.inputpath,
        probs_path=args.probspath,
        label_path=args.labelpath,
        save=save,
        save_path=args.savepath,
        load_path=args.loadpath,
    )
