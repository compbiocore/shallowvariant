# TO DO: GET data from files and turn it into usable things
import tensorflow as tf
import numpy as np
import os

input_spec = {
        'label': tf.io.FixedLenFeature((), tf.int64),
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'image/format': tf.io.FixedLenFeature((), tf.string),
        # 'image/shape': tf.io.FixedLenFeature((), tf.int64),
        'variant/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'variant_type': tf.io.FixedLenFeature((), tf.int64),
        # 'sequencing_type': tf.io.FixedLenFeature([], tf.int64),
    }

def _parse_inputs_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(example_proto, input_spec)
    image = parsed['image/encoded']
    # If the input is empty there won't be a tensor_shape.
    image = tf.reshape(tf.io.decode_raw(image, tf.uint8), [100, 221, 6])
    image = tf.dtypes.cast(image, tf.float32)

    return image

def _parse_labels_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(example_proto, input_spec)
    val = parsed['label']
    # If the input is empty there won't be a tensor_shape.
    encoded = tf.one_hot([val], depth=3)

    return encoded

def get_data_train(input_path, probs_path):
    # get pile-up images
    input_files = [os.path.join(input_path, x) for x in os.listdir(input_path)]
    inputs = tf.data.TFRecordDataset(input_files, compression_type="GZIP")
    parsed_inputs = inputs.map(_parse_inputs_function)

    # get probabilities from tsv
    probs = []
    probs_files = [os.path.join(probs_path, x) for x in os.listdir(probs_path)]
    for file in probs_files:
        with open(file, "r") as f:
            probs.extend(map(float, f.read().split()))

    p = np.array(probs)
    p = np.reshape(p, [-1, 3])

    # n = p.shape[0]

    p = tf.data.Dataset.from_tensor_slices(p)

    return tf.data.Dataset.zip((parsed_inputs, p))

def get_data_eval(input_path, label_path):
    # get pile-up images
    input_files = [os.path.join(input_path, x) for x in os.listdir(input_path)]
    inputs = tf.data.TFRecordDataset(input_files, compression_type="GZIP")
    parsed_inputs = inputs.map(_parse_inputs_function)

    label_files = [os.path.join(input_path, x) for x in os.listdir(label_path)]
    labels = tf.data.TFRecordDataset(label_files, compression_type="GZIP")
    parsed_labels = labels.map(_parse_labels_function)

    return tf.data.Dataset.zip((parsed_inputs, parsed_labels))


# Example of getting data and using it
# https://github.com/google/deepvariant/blob/r0.9/docs/visualizing_examples.ipynb
#
# Currently three things in the spec from the latest deep variant code are commented out, because they weren't in the test run data
#
# Can pass an array of file_paths instaed of one

# https://github.com/protocolbuffers/protobuf
