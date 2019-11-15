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

# Notes on running DeepVariant

A working script is at [./script.sh](script.sh). It is currently running on a small test dataset, and is based off of [this tutorial](https://cloud.google.com/life-sciences/docs/tutorials/deepvariant). The tutorial also shows how to optimize GCP so that it runs in a couple hours at the cost of $3-4 on a 50x WGS dataset with a GPU rather than ~5 hours with CPUs. In the interest of time and if we can get the credits we will be using the GPU setup. Once the test run is done we will have an idea of what the output files look like, which we should be able to access with the command

```
gsutil ls gs://initialrun/
```

It should also be viewable inside the `initialrun` bucket on the GCP interface. We are specifically hoping to be able to look at the `TFRecord` files that should be outputted by the `call_variants` portion of DeepVariant, as described [here](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-details.md). Then we can move onto details of implementing the child model, as well as running optimized DeepVariant on a larger dataset to get the `TFRecord` files that will serve as input into the child model.

To check on the status of the run:

```
gcloud alpha genomics operations wait projects/deepvariant-259121/operations/18024102458209955812
```

Also see: [more details on DeepVariant quick start](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-quick-start.md).
