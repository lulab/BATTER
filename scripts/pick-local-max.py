#!/usr/bin/env python
import argparse
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description='pick best interval from overlapped ones')
    parser.add_argument('--input', '-i',required=True,help="input intervals")
    parser.add_argument('--output','-o',required=True,help="output intervals")
    args = parser.parse_args()


    logger.info(f"load intervals from {args.input} ...")
    logger.info(f"picked intervals will be saved to {args.output} .")
    fin = open(args.input)
    seq_id = ""
    last_seq_id, last_start, last_end = "", -1 ,-1
    fout = open(args.output,"w")
    cached_ivs = []
    cached_scores = []
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
                    print(mseq_id, mstart, mend, ".", round(mscore,3), mstrand, file=fout, sep="\t")
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
            print(mseq_id, mstart, mend, ".", round(mscore,3), mstrand, file=fout, sep="\t")
    fin.close()
    fout.close()



if __name__ == "__main__":
    main()
