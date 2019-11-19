# TO DO: GET data from files and turn it into usable things
import tensorflow as tf

spec = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'image/format': tf.io.FixedLenFeature((), tf.string),
        # 'image/shape': tf.io.FixedLenFeature((), tf.int64),
        'variant/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
        # 'variant_type': tf.io.FixedLenFeature((), tf.int64),
        # 'sequencing_type': tf.io.FixedLenFeature([], tf.int64),
    }

def _parse_example_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(example_proto, spec)
    image = parsed['image/encoded']
    # If the input is empty there won't be a tensor_shape.
    image = tf.reshape(tf.io.decode_raw(image, tf.uint8), [100, 221, 6])
    image = tf.dtypes.cast(image, tf.int32)

    return image

def get_data(file_path):
    examples = tf.data.TFRecordDataset(file_path, compression_type="GZIP")
    parsed_examples = examples.map(_parse_example_function)

    return parsed_examples


# Example of getting data and using it
# https://github.com/google/deepvariant/blob/r0.9/docs/visualizing_examples.ipynb
#
# Currently three things in the spec from the latest deep variant code are commented out, because they weren't in the test run data
#
# Can pass an array of file_paths instaed of one
