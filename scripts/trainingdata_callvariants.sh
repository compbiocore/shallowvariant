BIN_VERSION=0.7.2
MODEL=gs://deepvariant/models/DeepVariant/0.7.2/DeepVariant-inception_v3-0.7.2+data-wgs_standard
time sudo docker run \
    -v "/home/august_guang/out:/out" \
    gcr.io/deepvariant-docker/deepvariant:"${BIN_VERSION}" \
    /opt/deepvariant/bin/call_variants \
  --examples /out/train1.tfrecord.gz \
  --outfile /out/train1_candidates.tfrecord.gz \
  --checkpoint ${MODEL}/model.ckpt