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
