from deepvariant.protos import deepvariant_pb2
from third_party.nucleus.util import io_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i")
parser.add_argument("--outfile", "-o")

def read_write_call_variant_output(infile, outfile):
    probs = []
    for record in io_utils.read_tfrecords(infile, deepvariant_pb2.CallVariantsOutput):
        probs.append(record.genotype_probabilities)

    f = open(outfile, "w")

    f.write("\t".join(probs))

    f.close()

if __name__ == "__main__":
    args = parser.parse_args()
    read_write_call_variant_output(args.infile, args.outfile)
