# BATTER



## Introduction

### Background in bacteria transcription termination
- Two mechanisms, rho independent terminations (RITs) and rho dependent terminations (RDTs), contribute to transcription termination in bacteria. RITs are relatively easy to predict, but current methods are mostly optimized for a few model organisms. Rho protein binds to RUT (Rho utilization) sites and induces RDT. RUTs lack compact sequence pattern, and are hard to predict.
- The 3' end of functional transcript is often associated with a stem loop structure, which is thought to prevent exonuclease degration and maintain transcript integrity. This fact holds for 3' ends generated by both RITs and RDTs. 
  - For RIT, rho independent terminator itself serve as this protective stem loop, which marks the transcript 3' end.
  - For RDT, RUT site, where transcription termination actually happens, often locates at downstream of transcript 3' ends marked by the protective stem loop.  
  - **The prediction of primary transcript 3' ends is a easier task**.

### When to use BATTER
- BATTER (**BA**cteria **T**ranscript **T**hree prime **E**nd **R**ecognizer) is a tool for bacteria transcription termination prediction, it was designed to predict 3' ends of primary transcript (both RIT and RDT) across diverse species in bacteria domain. 
- If you want to predict 3' ends of functional transcripts generated by primary termination events, or RIT-based regulatory premature termination events, consider the `batter-tpe` program in `BATTER`. `batter-tpe` is a machine learning model based on BERT-CRF architecture.
- If you want to search RUT-like sequences, consider the `batter-rut` program in `BATTER`. `batter-rut` use a rule based algorithm to search RUT patterns, that is properly spaced YC (UC/CC) dimers. In general, `batter-rut` has a higher false positive rate  but it can provide useful insights when considering specific genomic regions, such as downstream sequence of 3' ends predicted by `batter-tpe`.

## Installation

### Dependency

- The following python packages are required:
  - [pytorch](https://pytorch.org/): test on version `1.7.0+cu110`, other version should work
  - [transformers](https://huggingface.co/docs/transformers/index): version `4.18.0`
  - [pyfaidx](https://pythonhosted.org/pyfaidx/): test on version `0.7.1`

- The following tools are optional:
  - [bedtools](https://bedtools.readthedocs.io/), for annotation of predictions

- We recommend using [miniforge](https://github.com/conda-forge/miniforge) to install the dependency. 
- After installing miniforge (please refer to instruction [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) ), you can create a new environment and activate it:

```bash
# create a new environment called batter-env
mamba create -n batter-env
# activate the environment
mamba activate batter-env
```

- Then you can install the dependencies:

```bash
# install required packages:
mamba install -c pytorch pytorch==1.7.0
mamba install -c conda-forge transformers==4.18.0
mamba install -c bioconda pyfaidx==0.7.1

# install optional packages if you need them:
mamba install -c bioconda bedtools
```

### Download

- After installing the dependencies, you can install BATTER by simply cloning this repo:

```bash
git clone https://github.com/uaauaguga/batter.git 
```

## Usage

- `BATTER` takes bacteria genome sequences (can be contigs or complete/draft genomes) as input, and produces predicted terminator coordinate and strand information with confidence scores in bed format
- Two modules, `batter-tpe` and `batter-rut` are available. 
- Here we take Escherichia coli str. K-12 substr. MG1655 (https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/) as an example. 

### batter-tpe

- Genome wide prediction of transcript 3' ends The following command takes around 2 min to finish on a nvidia V100 GPU.

```bash
# take Escherichia coli str. K-12 substr. MG1655 GCF_000005845.2
# The default batch size if 256. If the GPU memory is limitted, plase use a smaller batch size, eg. 64
# The following command takes around 2 min to finish on a nvidia V100 GPU.
scripts/batter-tpe --fasta examples/e.coli/genome.fna --output examples/e.coli/TPE.bed --device cuda:0
```

- The output has 7 fields (the 1-6 columns are in standard bed format)
  1. contig / chromosome name
  2. start position
  3. end position
  4. ID of predicted 3' end
  5. score of the prediction
  6. strand
  7. estimated number of false positive per-kilobase

- Notes
  - BATTER scans both strand by default, if you only need to scan top strand, use `-nrc` option.
  - If you want to keep temporary file, use `--keep-temp`/`-kt` option. You can also specify path of temporary file with parameter `--tmp-file`
  - If more efficient scanning (at cost of lower sensitivity) is desired, you can increase the step size (100 nt by default) for scanning 

### batter-rut
- Prediction of putative RUT sites in specified regions

```bash
# predict 
scripts/batter-rut -i examples/e.coli/genome.fna -ivs examples/e.coli/TPE.bed --left-slop 100 --right-slop 200 --output examples/e.coli/TPE.RUT.bed
```

- The output has 7 fields (the 1-6 columns are in standard bed format)
  1. contig / chromosome name
  2. start position
  3. end position
  4. ID of predicted 3' end
  5. score of the prediction
  6. strand
  7. score at candidate identification stage
  8. estimated number of false positive per-kilobase

### Annotation

- Computational annotation of prokaryote genome, especially annotation of proteining coding genes, is relatively reliable, and you can annotate predicted terminators with its relative position to protein coding gene
- You can download bacteria genome annotation from ncbi, or perform annotation using tools like [prokka](https://github.com/tseemann/prokka). You'll get a file in gff format.

- Convert CDS annotation to bed format

```bash
# convert gff format to bed format
scripts/gff2bed.py --gff examples/e.coli/genes.gff --bed examples/e.coli/genes.bed --feature CDS --name ID
# make sure the genes are sorted by coordinate
sort -k1,1 -k2,2n -o examples/e.coli/genes.bed examples/e.coli/genes.bed
``` 
- Annotate predicted terminators with protein coding genes

```bash
scripts/annotate-intervals.py --gene examples/e.coli/genes.bed --bed examples/e.coli/TPE.bed --contig examples/e.coli/genome.fna.fai --output examples/e.coli/TPE.annotated.bed
scripts/annotate-intervals.py --gene examples/e.coli/genes.bed --bed examples/e.coli/TPE.RUT.bed --contig examples/e.coli/genome.fna.fai --output examples/e.coli/TPE.RUT.annotated.bed
```
- Explanation  of the output :
  1. contig id    
  2. start     
  3. end  
  4. ID
  5. score
  6. strand
  7. direction
  8. relative distance to upstream gene and to downstream gene, seperated by comma
  9. whether strand is same (concordant) or different (discordant) with associated gene
  10. location relative to CDS, one of leader, downstream, genic, or intergenic
  11. fraction overlap with CDS

## Citation

Yunfan, J., et al., BATTER: Accurate Prediction of Rho-dependent and Rho-independent Transcription Terminators in Metagenomes. bioRxiv, 2023: p. 2023.10.02.560326.
