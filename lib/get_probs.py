"""
This file is copied into the docker image, where it can be used to convert the outputs of call_variants to a tsv of probabilitites.

From /shallowvariant:
docker build -t get_probs .
docker run -it -v `pwd`:/input get_probs
python get_probs.py -i /input/data/labels/<infile> -o /input/data/labels/<outfile>
"""

from deepvariant.protos import deepvariant_pb2
from third_party.nucleus.util import io_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i")
parser.add_argument("--outfile", "-o")

def read_write_call_variant_output(infile, outfile):
    f = open(outfile, "w")
    for record in io_utils.read_tfrecords(infile, deepvariant_pb2.CallVariantsOutput):
        probs = list(record.genotype_probabilities)
        f.write("\t".join(map(str, probs)))
        f.write("\n")

    f.close()

if __name__ == "__main__":
    args = parser.parse_args()
    read_write_call_variant_output(args.infile, args.outfile)
