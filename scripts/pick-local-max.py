#!/usr/bin/env python
import argparse
import numpy as np
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("select intervals")

def merge_intervals(cached_scores, cached_ivs):
    cached_scores_by_strand = {"+":[],"-":[]}
    cached_iv_by_strand = {"+":[],"-":[]}
    for i in range(len(cached_scores)):
        seq_id, start, end, strand = cached_ivs[i]
        cached_scores_by_strand[strand].append(cached_scores[i])
        cached_iv_by_strand[strand].append((seq_id, start, end))
    score_by_strand, iv_by_strand = {}, {}

    for strand in "+-":
        if len(cached_scores_by_strand[strand]) == 0:
            continue
        i = np.argmax(cached_scores_by_strand[strand])
        score_by_strand[strand] = cached_scores_by_strand[strand][i]
        iv_by_strand[strand] = cached_iv_by_strand[strand][i]
    return score_by_strand, iv_by_strand

exedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description='pick best interval from overlapped ones')
    parser.add_argument('--input', '-i',required=True,help="input intervals")
    parser.add_argument('--output','-o',required=True,help="output intervals")
    parser.add_argument('--lookup','-l',required=True,help="bin lookup")
    parser.add_argument('--fprs','-f',default="model/TPE-FPR-by-clusters.txt",help="fprs lookup table")
    args = parser.parse_args()

    logger.info("Load FPR table ...")
    fprs = pd.read_csv(os.path.join(exedir,args.fprs),sep="\t",index_col=0)
    lut = {}
    with open(args.lookup) as f:
        for line in f:
            seq_id, binidx = line.strip().split("\t")
            lut[seq_id] = binidx

    logger.info(f"load intervals from {args.input} ...")
    logger.info(f"picked intervals will be saved to {args.output} .")
    fin = open(args.input)
    seq_id = ""
    last_seq_id, last_start, last_end = "", -1 ,-1
    fout = open(args.output,"w")
    cached_ivs = []
    cached_scores = []
    idx = 0
    for line in fin:
        seq_id, start, end, _, scores, strand = line.strip().split("\t")
        start, end = int(start), int(end)
        scores = np.array(scores.split(",")).astype(float)
        score = np.mean(scores)
        if ((start > last_end) and (last_seq_id == seq_id)) or (last_seq_id != seq_id):
            # current entry does not overlap with previous one
            # save cached entries
            if len(cached_ivs) > 0:
                # group predictions by strand
                score_by_strand, iv_by_strand = merge_intervals(cached_scores, cached_ivs)
                for mstrand in "+-":
                    if mstrand not in score_by_strand:
                        continue
                    mseq_id, mstart, mend  = iv_by_strand[mstrand]
                    mscore = score_by_strand[mstrand]
                    mscore = round(mscore,3)
                    level = int(mscore*1000)
                    if level >= 486:                 
                        fpr = round(fprs.loc[level,lut[mseq_id]],5)  
                    else:
                        fpr = np.nan
                    print(mseq_id, mstart, mend, "TPE" + str(idx).zfill(8), mscore, mstrand, fpr, file=fout, sep="\t")
                    idx += 1
                # update the cache
                cached_ivs, cached_scores = [], []
        cached_ivs.append((seq_id, start, end, strand))
        cached_scores.append(score)
        last_seq_id, last_start, last_end = seq_id, start, end

    if len(cached_ivs) > 0:
        score_by_strand, iv_by_strand = merge_intervals(cached_scores, cached_ivs)
        for mstrand in "+-":
            if mstrand not in score_by_strand:
                continue
            mseq_id, mstart, mend = iv_by_strand[mstrand]
            mscore = score_by_strand[strand]
            mscore = round(mscore,3)
            level = int(mscore*1000)
            if level >= 486:
                fpr = round(fprs.loc[level,lut[mseq_id]],5)
            else:
                fpr = np.nan
            print(mseq_id, mstart, mend, "TPE" + str(idx).zfill(8), mscore, mstrand, fpr, file=fout, sep="\t")
    fin.close()
    fout.close()



if __name__ == "__main__":
    main()
