# batter

- batter (**ba**cteria **t**ranscript **t**hree prime **e**nd **r**ecognizer) is a tool for bacteria transcription terminator prediction based on BERT-CRF model. 
- It aims to predict both rho dependent and rho independent terminator across diverse species in Bacteria domain.

## Installation

### dependency

- the following python packages are required
  - [pytorch](https://pytorch.org/): test on version `1.7.0+cu110`, other version should work
  - [transformers](https://huggingface.co/docs/transformers/index): test on version `4.18.0`
  - [pyfaidx](https://pythonhosted.org/pyfaidx/): test on version `0.7.1`
  - [numpy](https://numpy.org/)

- optional
  - [lightgbm](https://lightgbm.readthedocs.io/), for predicting score cutoff given tetra-mer frequency
  - [bedtools](https://bedtools.readthedocs.io/), for evaluating genomic distribution of predicted terminator

### Installation

- Simply clone this repo:

```{bash}
git clone https://github.com/uaauaguga/batter.git 
```


## usage

- `batter` takes bacteria genome sequence (can be contig or complete/draft genome) as input, and produces predicted terminator coordinate and strand information with confidence scores in bed format

### Inference

- Here we take scanning S.aureus genome [GCF_000013425.1](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/013/425/GCF_000013425.1_ASM1342v1/) as an example. 

- batter scans top strand by default, for genome scanning, searching both top and bottom strand is desired, here we use `--reverse-complement`/`-rc` option. The following command takes around 2 min to finish on a nvidia V100 GPU.

```bash
# the default batch size if 256. If the GPU memory is limitted, plase use a smaller batch size, eg. 64
scripts/batter --fasta examples/S.aureus/genome.fna --output examples/S.aureus.bed --device cuda:0 -rc
```

- The output is in bed format
  - field 0: contig / chromosome name
  - field 1: start position
  - field 2: end position
  - field 3: omited 
  - field 4: score of the prediction
  - field 5: strand

 
- if you want to keep temperorary file, use `--keep-temp`/`-kt` option. You can also specify path of temporary file with parameter `--tmp-file`
 
```bash
 scripts/batter --fasta examples/S.aureus/genome.fna --output examples/S.aureus.bed --device cuda:0 -rc -kt
```

- if more efficient scanning (at cost of lower sensitivity) is desired, you can increase the step size (100 nt by default) for scanning 

```bash
 # the following command take ~1 min on a V100 GPU, but produce less prediction
 scripts/batter --fasta examples/S.aureus/genome.fna --output examples/S.aureus.250.bed --device cuda:0 -rc --stride 250
```

### Model calibration

 The output of batter are genomic intervals associated with a score. The predictions are conservative, but for low GC-content bacteria species, the false positive rate tend to be relatively higher. There are several choices to determine which cutoff to use. 

- A cutoff of 0.5 is generally safe.
- The mean cutoffs of species with different genomic GC content that achieve a false positive rate (FPR) of 0.1/KB are listed as follow:

```bash
# you may check GC content with the following command. many other tools does same thing
scripts/kmer-frequency-fitter.py -i examples/S.aureus/genome.fna -o examples/S.aureus/genome.nuc.freq -k 1
# kmer	A	C	G	T
# genome	33.27063	16.51051	16.35686	33.85906
```

| GC content | cutoff |
| ---------- | ------ |  
| 20%-30%    | 0.547  |
| 30%-40%    | 0.533  |
| 40%-50%    | 0.511  |
| 50%-60%    | 0.496  |
| 60%-70%    | 0.491  |
| 70%-80%    | 0.489  |

- Providing your genome, calculate tetra-mer frequency (TNF), and predict cutoff at FPR of 0.1 with a pretrained model (this feature depends on [lightgbm](https://lightgbm.readthedocs.io/) package) 

```bash
# calculate TNF distribution 
scripts/kmer-frequency-fitter.py -i examples/S.aureus/genome.fna -o examples/S.aureus/genome.TNF
# predict 
scripts/tnf-score-cutoff-predictor.py -tnf examples/S.aureus/genome.TNF --scores examples/S.aureus/genome.cutoff 
# cat examples/S.aureus/genome.cutoff
# genome id	score
# genome	0.5252
```

- Of course you can simulate a background sequence set, and determine the cutoff yourself.

- Then we can filter the predicted terminators with a cutoff specified by one of the above approaches:

```bash
cat examples/S.aureus.bed | awk '$5>0.5252{print}' > examples/S.aureus.filtered.bed
```

### Annotation

- Computational annotation of prokaryote genome, especially annotation of proteining coding genes, is relatively reliable, and you can annotate predicted terminators with its relative position to protein coding gene
- You can download bacteria genome annotation from ncbi, or perform annotation using tools like [prokka](https://github.com/tseemann/prokka). You'll get a file in gff format.

- Convert CDS annotation to bed format

```bash
scripts/gff2bed.py --gff examples/S.aureus/annotation.gff --bed examples/S.aureus/annotation.bed --feature CDS --name ID
``` 
- Annotate predicted terminators with protein coding genes

```bash
scripts/annotate-intervals.py --gene examples/S.aureus/annotation.bed --bed examples/S.aureus.250.filtered.bed --contig examples/S.aureus/genome.fna.fai --output examples/S.aureus.250.filtered.annotated.bed
```

- Example output

```text
NC_007795.1	1885	1942	.	0.545	+	>|>	8,214	concordant	downstream	cds-YP_498609.1|cds-YP_498610.1
NC_007795.1	3292	3350	.	0.851	+	>|>	4,320	concordant	downstream	cds-YP_498610.1|cds-YP_498611.1
NC_007795.1	3299	3343	.	0.545	-	>|>	11,327	discordant	downstream	cds-YP_498610.1|cds-YP_498611.1
NC_007795.1	9663	9715	.	0.894	+	genic	0	concordant	gene:3'	cds-YP_498614.1
#    0            1       2     3          4    5        6      7            8               9       10
# 0: contig id    
# 1: start     
# 2: end  
# 3: omit
# 4: score
# 5: strand
# 6: direction
# 7: relative distance to upstream gene and to downstream gene, seperated by comma
# 8: whether strand is same (concordant) or different (discordant) with associated gene
# 9: location relative to CDS, one of leader, downstream, genic, or intergenic
# 10: id of upstream gene and downstream gene
```
