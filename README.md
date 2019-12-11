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
