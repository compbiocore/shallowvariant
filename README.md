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

# Running the Model

For training, the probabilities first need to be extracted from the called variants outputted from deep variant

To do this:

```
docker build -t get_prob .
docker run -it -v `pwd`:/input get_probs
python get_probs.py -i /input/data/call_variants/<infile> -o /input/data/probs/<outfile>
```

The probabilities will now be saved as a tab separated text file in the data folder.

To train the model:
`python -m src.shallowvariant` with the folowing args

* `--train` : sets model to training mode, otherwise test
* `--cnnsize`, `-c`: layer size for cnn layer, default 100
* `--learningrate`, `-r`: learning rate of the model, default .01
* `--inputpath`, `-i`: directory where the input examples (pileup images) are stored.  This should be a directory containing at least one `.tfrecord.gz` from the `make_examples` output of DeepVariant
* `--probspath`, `-p`: directory where the probabilities from the docker-based step above are stored.
* `--labelpath`, `-l`: directory where the examples with labels are stored.  This should be a directory containing at least one `.tfrecord.gz` fromt he `make_examples` output of DeepVariant in `TRAIN` mode
* `--savepath`, `-s`: file path where the model should be saved, if this is set, the model will save itself after training
* `--loadpath`, `-m`: file path where the model should be loaded from

Example for training:
```
python -m src.shallowvariant -i data/real_labels -p data/probs -s data/my_model.h5 -r .001 -c 128
```

Example for testing:
```
python -m src.shallowvariant -i data/real_labels -l data/real_labels -m data/my_model.h5
```

# Regenerating training data and testing labels

To regenerate the training data and labels, after spinning up a GCP instance (can be any number of cores), copy and paste everything in `scripts/trainingdata.sh`. This will run `make_examples` in training mode, which means generating pileup images and labels. Once that is finished running, copy and paste everything in `scripts/trainingdata_callvariants.sh`. This will run `call_variants`, which outputs probabilities from the pileup images. WARNING: this will likely take 24+ hours.

# Progress

## 11/15/19

A working script for DeepVariant is at [script.sh](./script.sh). It ran on a small test dataset, and is based off of [this tutorial](https://cloud.google.com/life-sciences/docs/tutorials/deepvariant). The tutorial also shows how to optimize GCP so that it runs in a couple hours at the cost of $3-4 on a 50x WGS dataset with a GPU rather than ~5 hours with CPUs. In the interest of time and if we can get the credits we will be using the GPU setup. We looked at the `TFRecord` files output by the `call_variants` portion of DeepVariant, as described [here](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-details.md), as we were planning to use them, as input for our child model. We were hoping to be able to look at them with base TensorFlow, but we found out it requires the [Nucleus](https://github.com/google/nucleus) library to be able to view them. Additionally Nucleus only works on Linux, and we both have macOS.

```
gsutil ls gs://initialrun/
```

Also see: [more details on DeepVariant quick start](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-quick-start.md).

We also have an initial implementation for the child model, which is a CNN layer and a dense layer in [src](./src), along with a script to run the model. There is also a skeleton for `get_data()`, although we are waiting to see how to use the `TFRecord` files as input.

## 11/21/19

 * The `TFRecord` objects outputted by `call_variants` are actually storing many ProtoBufs that can only be opened with DeepVariant-specific code which also relies on Nucleus, and DeepVariant has to be built for us to be able to extract the objects. Mary has written a Dockerfile that pulls DeepVariant and converts the ProtoBufs into a `tsv` file that is the actual probabilities. Validation: the number of probabilities matches the number of examples (82). We do not know if the order matches, but we hope so.
 * The `TFRecord` examples which are the output of `make_examples` are just `TFExamples` which is a `ProtoBuf` with an undocumented dictionary on top. The dictionary lives in the `DeepVariant` code.
 * The `DeepVariant` version we are going to stick to is .7.2, as there are significant changes in current versions that also change the `TFRecord` objects. This may prove a problem for maintainability as if we wanted to use more current versions we would have to invest re-engineering work.
 * Right now, `preprocess` reads files for input, parses images, gets the images into the right shape, and outputs those as inputs for the model. Still need to work on labels. Also need to figure out split and shuffling.
 * August got the true labels for our test dataset by running DeepVariant in `train` mode, which means turning on the flags `true_vcf` and `confident_regions` with [data from their quickstart](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-quick-start.md).
 * August is going to work on getting out the true labels, of which there should be 82. Mary is going to continue working on the model.

## 12/6/19

 * We moved onto running the model on a large training data set. The data was generated using DeepVariant's `make_examples` and then extracted.
 * Although the data comes from GenieInABottle, in practice we just accessed the data already present in the deepvariant bucket on GCP, which can be accessed with `gsutil ls gs://deepvariant`. We made use of the data from the Training Case Study, the [Case Study](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-case-study.md).

# Next steps

 * Continue working on `get_data()` based off of the `TFRecord` files. We will have to split this data ourselves, but the ingestion of the files should be straightforward based off of DeepVariant.
 * Implement getting the labels from the `TFRecord` files. This could possibly be in `get_data()`, or in a different function.
 * Run the child model to make sure it doesn't crash.
 * If we get the model working (and credits), run DeepVariant on GCP on a larger dataset to get initial input. This will take a couple hours, so we will want to monitor the run and shut off the instance after.
 * Our larger dataset will come from [gold standard data from GIAB](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh37/). We can match true labels to data based on DeepVariant's documentation about how [they trained their model](https://github.com/google/deepvariant/blob/r0.9/docs/deepvariant-details-training-data.md).
