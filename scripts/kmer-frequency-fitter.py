#!/usr/bin/env python
from collections import defaultdict
import argparse
import logging
from itertools import product
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("get kmer frequency distribution of given fasta file")

def main():
    parser = argparse.ArgumentParser(description='count k-mer frequency')
    parser.add_argument('--input',  '-i', type=str, required=True, help='input fasta')
    parser.add_argument('--output', '-o', type=str, required=True, help='output frequency table')
    parser.add_argument('--word-size', '-k', type=int, default=4, help='word size to use')
    args = parser.parse_args()

    kmers = ["".join(c) for c in product(*["ACGT"]*args.word_size)]
    
    logger.info("load sequence ...")
    sequences = defaultdict(str)
    with open(args.input) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line.strip()[1:].split(" ")[0]
            sequences[seq_id] += line.strip()
    

    logger.info(f"count {args.word_size} mer ...")
    counter = defaultdict(int)
    total = 0
    for seq_id in sequences:
        sequence = sequences[seq_id]
        for i in range(len(sequence)-args.word_size):
            kmer = sequence[i:i+args.word_size]
            counter[kmer] += 1
            total += 1

    logger.info(f"saving results to {args.output} ...")
    fout = open(args.output,"w")
    fractions = []
    for kmer in kmers:
        fraction = 100*counter[kmer]/total
        fractions.append(round(fraction,5))
    print("kmer", *kmers, sep="\t",file=fout)
    sample_id = ".".join(args.input.split("/")[-1].split(".")[:-1])
    print(sample_id, *fractions, sep="\t", file=fout)
    fout.close()

    logger.info("all done .")


if __name__ == "__main__":
    main()



            
            
