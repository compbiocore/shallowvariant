import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf
from tensorflow.core.example import example_pb2

def get_bytes_list(example, key):
    """Returns the bytes-list corresponding to a key in a tf.Example."""
    return example.features.feature[key].bytes_list.value


def get_label(example):
    """Returns the label(class) of the example. Expected values are 0 (HOM-REF), 1(HET), 2(HOM-ALT)"""
    return example['label'].numpy()

def imshows(images, fname, labels=None, n=None, scale=10, axis='off', **kwargs):
    """Plots a list of images in a row.

    Args:
        images: List of image arrays.
        labels: List of str or None. Image titles.
        n: int or None. How many images in the list to display. If this is None,
          display all of them.
        scale: int. How big each image is.
        axis: str. How to plot each image.
        **kwargs: Keyword arguments for Axes.imshow.
    Returns:
        None.
    """
    n = len(images) if (n is None) else n
    with sns.axes_style('white'):
        _, axs = plt.subplots(1, n, figsize=(n * scale, scale))
        for i in range(n):
            axs[i].imshow(images[i], **kwargs)
            axs[i].axis(axis)
            if labels:
                axs[i].set_title(labels[i])
        plt.savefig(fname)

def visualize_example(example, fname):
    image = example['image/encoded'].numpy()
    # Read in the image bytestring and its shape: [100, 221, 6].
    shape = (100, 221, 6)
    # Parse the bytestring and reshape to an image.
    image = np.frombuffer(image, np.uint8).reshape(shape)
    # Split the tensor by its channels dimension and plot.
    channels = [image[:, :, i] for i in range(shape[-1])]
    # Prepend an image: RGBA image reconstructed from the 6-channels
    channels.insert(0, channels_to_rgb(channels))
    titles = ["reconstructed RGBA (label=%s)" % get_label(example),
              "read base", "base quality", "mapping quality", "strand",
              "supports variant", "supports reference"]
    imshows(channels, fname, titles, axis="image", scale=5)

def channels_to_rgb(channels):
    # Reconstruct the original channels
    base = channels[0]
    qual = np.minimum(channels[1], channels[2])
    strand = channels[3]
    alpha = np.multiply(
        channels[4] / 254.0,
        channels[5] / 254.0)
    return np.multiply(
        np.stack([base, qual, strand]),
        alpha).astype(np.uint8).transpose([1, 2, 0])

example_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'variant/encoded': tf.io.FixedLenFeature([], tf.string),
}

def _parse_example_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, example_description)

# This tfrecord comes from HG002 PFDA data. I ran make_examples in training mode so we also have the labels.
src = '/Users/aguang/Classes/CSCI1470/shallowvariant/data/training_set2.with_label.tfrecord.gz'

examples = tf.data.TFRecordDataset(src, compression_type="GZIP")
parsed_examples = examples.map(_parse_example_function)

i=0
for example in parsed_examples.take(5):
    visualize_example(example, "shallowimg_%s.pdf" % i)
    i = i + 1