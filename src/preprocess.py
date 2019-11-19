# TO DO: GET data from files and turn it into usable things
import tensorflow as tf

spec = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'variant/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'variant_type': tf.io.FixedLenFeature((), tf.int64),
        # 'sequencing_type': tf.io.FixedLenFeature([], tf.int64),
    }

def _parse_example_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, spec)

def get_data(file_path):
    examples = tf.data.TFRecordDataset(file_path, compression_type="GZIP")
    parsed_examples = examples.map(_parse_example_function)

    return parsed_examples
