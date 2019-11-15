# ShallowVariant

ShallowVariant is an attempt to build a child model for the variant caller DeepVariant (for CSCI1470)

# Description

.vcf files are turned into Pileup images using the DeepVariant pre-processing pipeline.

The images are then passed to DeepVariant to get the predicted probabilities of:
* homozygous - reference
* heterozygous
* homozygous - alt (variant)

These probabilities are then used as the ground truth labels for training ShallowVariant.

ShallowVariant takes a Pilup image as it's input and returns the same probabilities as DeepVariant

# Progress

## 11/15/19

A working script for DeepVariant is at [script.sh](./script.sh). It ran on a small test dataset, and is based off of [this tutorial](https://cloud.google.com/life-sciences/docs/tutorials/deepvariant). The tutorial also shows how to optimize GCP so that it runs in a couple hours at the cost of $3-4 on a 50x WGS dataset with a GPU rather than ~5 hours with CPUs. In the interest of time and if we can get the credits we will be using the GPU setup. We looked at the `TFRecord` files output by the `call_variants` portion of DeepVariant, as described [here](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-details.md), as we were planning to use them, as input for our child model. We were hoping to be able to look at them with base TensorFlow, but we found out it requires the [Nucleus](https://github.com/google/nucleus) library to be able to view them. Additionally Nucleus only works on Linux, and we both have macOS.

```
gsutil ls gs://initialrun/
```

Also see: [more details on DeepVariant quick start](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-quick-start.md).

We also have an initial implementation for the child model, which is a CNN layer and a dense layer in [src](./src), along with a script to run the model. There is also a skeleton for `get_data()`, although we are waiting to see how to use the `TFRecord` files as input.

# Next steps

 * Spin up a Docker image with Linux and install [Nucleus](https://github.com/google/nucleus) on it, then open the `TFRecord` files using the library to see what they look like. They should be probabilities.
 * Implement `get_data()` based off of the `TFRecord` files. We will have to split this data ourselves, but the ingestion of the files should be straightforward based off of DeepVariant.
 * Implement getting the labels from the `TFRecord` files. This could possibly be in `get_data()`, or in a different function.
 * Run the child model to make sure it doesn't crash.
 * If we get the model working (and credits), run DeepVariant on GCP on a larger dataset to get initial input. This will take a couple hours, so we will want to monitor the run and shut off the instance after.
