# TO DO: GET data from files and turn it into usable things
import tensorflow as tf
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

label_spec = {
        'label': tf.io.FixedLenFeature((), tf.int64),
        #'variant': tf.io.FixedLenFeature((), tf.string),
        #'probabilities': tf.io.FixedLenFeature((), tf.string),
    }

def _parse_inputs_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(example_proto, input_spec)
    image = parsed['image/encoded']
    # If the input is empty there won't be a tensor_shape.
    image = tf.reshape(tf.io.decode_raw(image, tf.uint8), [100, 221, 6])
    image = tf.dtypes.cast(image, tf.int32)
    return image

def _parse_labels_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above
    parsed = tf.io.parse_single_example(example_proto, label_spec)
    labels = parsed['label']
    parsed_labels = []
    for l in labels:
        parsed_labels.append(l.numpy())
    return parsed_labels

def get_data(input_path, label_path):
    input_files = [os.path.join(input_path, x) for x in os.listdir(input_path)]
    label_files = [os.path.join(label_path, x) for x in os.listdir(label_path)]
    inputs = tf.data.TFRecordDataset(input_files, compression_type="GZIP")
    labels = tf.data.TFRecordDataset(label_files, compression_type="GZIP")
    parsed_inputs = inputs.map(_parse_inputs_function)
    parsed_labels = labels.map(_parse_labels_function)
    return parsed_inputs, parsed_labels

# Example of getting data and using it
# https://github.com/google/deepvariant/blob/r0.9/docs/visualizing_examples.ipynb
#
# Currently three things in the spec from the latest deep variant code are commented out, because they weren't in the test run data
#
# Can pass an array of file_paths instaed of one

# https://github.com/protocolbuffers/protobuf
