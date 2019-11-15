#!/bin/bash
set -euo pipefail
# Set common settings.
PROJECT_ID=deepvariant-259121
OUTPUT_BUCKET=gs://initialrun
STAGING_FOLDER_NAME=initialrun
OUTPUT_FILE_NAME=output.vcf
# Model for calling whole genome sequencing data.
MODEL=gs://deepvariant/models/DeepVariant/0.7.2/DeepVariant-inception_v3-0.7.2+data-wgs_standard
IMAGE_VERSION=0.7.2
DOCKER_IMAGE=gcr.io/deepvariant-docker/deepvariant:"${IMAGE_VERSION}"
COMMAND="/opt/deepvariant_runner/bin/gcp_deepvariant_runner \
  --project ${PROJECT_ID} \
  --zones us-west1-* \
  --docker_image ${DOCKER_IMAGE} \
  --outfile ${OUTPUT_BUCKET}/${OUTPUT_FILE_NAME} \
  --staging ${OUTPUT_BUCKET}/${STAGING_FOLDER_NAME} \
  --model ${MODEL} \
  --bam gs://deepvariant/quickstart-testdata/NA12878_S1.chr20.10_10p1mb.bam \
  --ref gs://deepvariant/quickstart-testdata/ucsc.hg19.chr20.unittest.fasta.gz \
  --regions chr20:10,000,000-10,010,000 \
  --gcsfuse"
# Run the pipeline.
gcloud alpha genomics pipelines run \
    --project "${PROJECT_ID}" \
    --service-account-scopes="https://www.googleapis.com/auth/cloud-platform" \
    --logging "${OUTPUT_BUCKET}/${STAGING_FOLDER_NAME}/runner_logs_$(date +%Y%m%d_%H%M%S).log" \
    --regions us-west1 \
    --docker-image gcr.io/cloud-genomics-pipelines/gcp-deepvariant-runner \
    --command-line "${COMMAND}"