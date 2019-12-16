/opt/deepvariant/bin/make_examples --mode training \
  --reads /home/data/NA12878_S1.chr20.10_10p1mb.bam \
  --ref /home/data/ucsc.hg19.chr20.unittest.fasta.gz \
  --regions chr20:10,000,000-10,010,000 \
  --truth_variants /home/data/test_nist.b37_chr20_100kbp_at_10mb.vcf.gz \
  --confident_regions /home/data/test_nist.b37_chr20_100kbp_at_10mb.bed \
  --examples /home/data/training_set.with_label.tfrecord.gz

/opt/deepvariant/bin/make_examples --mode training \
  --reads /home/data/NA12878_S1.chr20.10_10p1mb.bam \
  --ref /home/data/ucsc.hg19.chr20.unittest.fasta.gz \
  --regions chr20:10,000,000-10,010,000 \
  --truth_variants /home/data/test_nist.b37_chr20_100kbp_at_10mb.vcf.gz \
  --confident_regions /home/data/test_nist.b37_chr20_100kbp_at_10mb.bed \
  --examples /home/data/training_set2.with_label.tfrecord.gz \
  --candidates /home/data/candidate_set.with_label.tfrecord.gz

/opt/deepvariant/bin/make_examples --mode training \
  --reads /home/data/NA12878_S1.chr20.10_10p1mb.bam \
  --ref /home/data/ucsc.hg19.chr20.unittest.fasta.gz \
  --truth_variants /home/data/test_nist.b37_chr20_100kbp_at_10mb.vcf.gz \
  --confident_regions /home/data/test_nist.b37_chr20_100kbp_at_10mb.bed \
  --examples /home/data/training_set.with_label.tfrecord@8.gz

/opt/deepvariant/bin/make_examples --mode training \
  --reads /home/data/NA12878_S1.chr20.10_10p1mb.bam \
  --ref /home/data/ucsc.hg19.chr20.unittest.fasta.gz \
  --regions 20:10,000,000-10,010,000 \
  --truth_variants /home/data/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf.gz \
  --confident_regions /home/data/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel.bed \
  --examples /home/data/hg001.tfrecord@8.gz