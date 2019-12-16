#!/bin/bash
set -euo pipefail
# Set common settings.
PROJECT_ID=deepvariant-259121
sudo apt-get install parallel


# COPYING DATA
gsutil cp gs://deepvariant/training-case-study/BGISEQ-HG001/* ./train1
# do not actually need all of the data in here
gsutil cp gs://deepvariant/case-study-testdata/* ./train2
gunzip ucsc_hg19.fa.gzata/* ./train2
mv ucsc_hg19.fa.fai ucsc_hg19.fa.gz.fai

OUTPUT_BUCKET=shallowtraining
STAGING_FOLDER_NAME=trainingdata
OUTPUT_FILE_NAME=output.vcf
INPUT_BUCKET=train1
TRUTH_VCF1=/${INPUT_BUCKET}/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer_chrs_FIXED.vcf.gz
TRUTH_BED1=/${INPUT_BUCKET}/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_chr.bed
TRAIN_BAM1=/${INPUT_BUCKET}/BGISEQ_PE100_NA12878.sorted.bam
TRAIN_FASTQ1=/${INPUT_BUCKET}/ucsc_hg19.fa

INPUT_BUCKET=train2
TRUTH_BED2=/${INPUT_BUCKET}/HG002_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed
TRUTH_VCF2=/${INPUT_BUCKET}/HG002_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-22_v.3.3.2_highconf_triophased.vcf.gz
TRAIN_BAM2=/${INPUT_BUCKET}/HG002_NIST_150bp_50x.bam
TRAIN_FASTQ2=/${INPUT_BUCKET}/hs37d5.fa.gz

BIN_VERSION=0.7.2

sudo docker run \
    -v "/home/august_guang/train1":"/train1" \
    -v "/home/august_guang/out:/out" \
    gcr.io/deepvariant-docker/deepvariant:"${BIN_VERSION}" \
    /opt/deepvariant/bin/make_examples \
  --mode training \
  --reads ${TRAIN_BAM1} \
  --ref ${TRAIN_FASTQ1} \
  --examples /out/train1.tfrecord.gz \
  --truth_variants ${TRUTH_VCF1} \
  --confident_regions ${TRUTH_BED1} \
  --exclude_regions "chr20 chr21 chr22" \
  --sample_name hg001

LOG_DIR=logs
N_SHARDS=$(nproc)
( time seq 0 $(($(nproc)-1)) | \
  parallel --halt 2 --joblog "${LOG_DIR}/log" --res "${LOG_DIR}" \
    sudo docker run \
      -v "/home/august_guang/train2":"/train2" \
      -v "/home/august_guang/out:/out" \
      gcr.io/deepvariant-docker/deepvariant:"${BIN_VERSION}" \
      /opt/deepvariant/bin/make_examples \
      --mode training \
      --reads ${TRAIN_BAM2} \
      --ref ${TRAIN_FASTQ2} \
      --examples /out/train2.tfrecord@${N_SHARDS}.gz \
      --truth_variants ${TRUTH_VCF2} \
      --confident_regions ${TRUTH_BED2} \
      --task {} \
      --sample_name hg002 \
) >"${LOG_DIR}/train2.log" 2>&1

# if we wanted to exclude regions would have to enter these:
#>1 dna:chromosome chromosome:GRCh37:1:1:249250621:1
#>2 dna:chromosome chromosome:GRCh37:2:1:243199373:1
#>3 dna:chromosome chromosome:GRCh37:3:1:198022430:1

time sudo docker run \
    -v "/home/august_guang/out:/out" \
    gcr.io/deepvariant-docker/deepvariant:"${BIN_VERSION}" \
    /opt/deepvariant/bin/call_variants \
  --examples /out/train2.tfrecord.gz \
  --outfile /out/train2_candidates.tfrecord.gz