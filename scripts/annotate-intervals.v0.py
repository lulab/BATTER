#!/usr/bin/env python
import argparse
import subprocess
import os
import logging
import numpy as np
import io
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('annotate intervals')

def main():
    parser = argparse.ArgumentParser(description='annotate intervals according to genome annotatioon')
    parser.add_argument('--gene','-g',type=str, required=True, help='gene interval in bed format, strandness is required')
    parser.add_argument('--bed','-b',type=str, required=True, help="intervals in bed format")
    parser.add_argument('--output','-o',type=str, required=True, help="where to save the annotations")
    parser.add_argument('--contig', '-c', type=str, required=True, help="contig length")
    parser.add_argument('--offset', '-os', type=int, default = 32, help="distance inside CDS that are assigned as 5'/3'")
    parser.add_argument('--flank', '-fk', type=int, default = 128, help="distance outside that are assigned as leader/downstream")
    args = parser.parse_args()

    logger.info("Load intervals ...")
    strandness = {}
    scores = {}
    names = {}
    antisense_scores = {}
    antisense_names = {}
    with open(args.bed) as f:
        for line in f:
            fields = line.strip().split("\t")
            seq_id, start, end, name, score, strand = fields[:6]
            if strand == ".":
                continue
            if (seq_id, start, end) not in scores:
                strandness[(seq_id, start, end)] = strand
                scores[(seq_id, start, end)] = score
                names[(seq_id, start, end)] = name
            else:
                antisense_scores[(seq_id, start, end)] = score
                antisense_names[(seq_id, start, end)] = name

    up_distance = {}
    up_strandness = {}
    down_distance = {}
    down_strandness = {}
    up_gene_ids, down_gene_ids = {}, {}

    logger.info("Get distance of up stream gene ...") 
    cmd = ["bedtools","closest","-D","a","-id","-a","-","-b",args.gene,"-g",args.contig]
    logger.info("running " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stdin=subprocess.PIPE)   

    tmp = {}
    for seq_id, start, end in strandness:
        if seq_id not in tmp:
            tmp[seq_id] = []
        try:
            tmp[seq_id].append((int(start), int(end)))
        except:
            continue
    lines = ""
    with open(args.contig) as f:
        for line in f:
            seq_id = line.strip().split("\t")[0]
            if seq_id not in tmp:
                continue
            ivs = sorted(tmp[seq_id])
            for start, end in ivs:
                lines += f"{seq_id}\t{start}\t{end}\n"

    lines = lines.encode()

    genic_location = {}
    for line in proc.communicate(lines)[0].decode().split("\n"):
        # each line is a interval, the last line is distance to closest upstream features
        line = line.strip()
        if len(line) == 0:
            continue
        fields = line.strip().split("\t")
        seq_id, start, end = fields[:3]
        distance = int(fields[-1])
        up_distance[(seq_id, start, end)] = distance
        # for features that overlap with cds, we only need to consider it ones
        if distance == 0:
            # NC_004719.1	61486	61508	NC_004719.1	61467	62430	WP_011109717.1	.	+	0
            tstart, tend = int(start), int(end) 
            gstart, gend = int(fields[4]), int(fields[5])
            if fields[-2] == "+":
                gue = gstart + args.offset
                gds = gend - args.offset
                if gue >= gds:
                    genic_location[(seq_id, start, end)] = "inside"
                else:
                    if tend < gue:
                        genic_location[(seq_id, start, end)] = "5'"
                    elif tstart > gds:
                        genic_location[(seq_id, start, end)] = "3'"
                    else:
                        genic_location[(seq_id, start, end)] = "inside"
            else:
                gue = gend - args.offset
                gds = gstart + args.offset
                if gue <= gds:
                    genic_location[(seq_id, start, end)] = "inside"
                else:
                    if tstart > gue:
                        genic_location[(seq_id, start, end)] = "5'"
                    elif tend < gds:
                        genic_location[(seq_id, start, end)] = "3'"
                    else:
                        genic_location[(seq_id, start, end)] = "inside"
        up_strandness[(seq_id, start, end)] = fields[-2]
        up_gene_ids[(seq_id, start, end)] = fields[-4]
    

    logger.info("Get distance to downstream gene ...")
    cmd = ["bedtools","closest","-D","a","-iu","-a","-","-b",args.gene,"-g",args.contig]
    logger.info("running " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stdin=subprocess.PIPE)

    for line in proc.communicate(lines)[0].decode().split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        fields = line.strip().split("\t")
        seq_id, start, end = fields[:3]
        # genic cases is already handled
        down_distance[(seq_id, start, end)] = int(fields[-1])
        down_strandness[(seq_id, start, end)] = fields[-2]
        down_gene_ids[(seq_id, start, end)] = fields[-4]
    logger.info("Summarizing results ...")

    proc.stdin.close()
    proc.wait()

    # 3*3 = 9 cases
    annotation_lut = {"++":">|>", "+-":">|<","+.":">|.",
                      "-+":"<|>", "--":"<|<","-.":"<|.",
                      ".+":".|>", ".-":".|<","..":".|."}
    fout = open(args.output,"w")
    for seq_id, start, end in strandness:
        if (seq_id, start, end) not in up_distance or (seq_id, start, end) not in down_distance:
            logger.warning(f"interval {seq_id} {start} {end} does not present, skip it .")
        ud, dd = up_distance[(seq_id, start, end)], down_distance[(seq_id, start, end)]
        us, ds = up_strandness[(seq_id, start, end)], down_strandness[(seq_id, start, end)]
        if (up_gene_ids[(seq_id, start, end)] == ".") and (ud == -1):
            ud = np.inf
        if (down_gene_ids[(seq_id, start, end)] == ".") and (dd == -1):
            dd = np.inf
        ud, dd = abs(ud), abs(dd)
        if ud == 0 or dd == 0 :
            # genic
            distance = 0
            annotation = "genic"
            gene_id = up_gene_ids[(seq_id, start, end)] if ud < 0 else down_gene_ids[(seq_id, start, end)]
        else:
            # intergenic
            annotation = annotation_lut[us+ds]
            gene_id = f"{up_gene_ids[(seq_id, start, end)]}|{down_gene_ids[(seq_id, start, end)]}"
            distance = f"{ud},{dd}"
        s = strandness[(seq_id, start, end)]
        if annotation == "genic":
            assert us == ds
            assignment = "gene"
            if s != us:
                conflict = "discordant"
            else:
                conflict = "concordant"
            assignment += ":" + genic_location[(seq_id, start, end)]
        else:
            if min(ud, dd) >= args.flank:
                # handle orphan cases
                assignment = "orphan" if (max(ud, dd) < 100000) else "unassigned"
                # annotate strandness to the nearest one
                if ((ud <= dd) and (s == us)) or ((ud >= dd) and (s == ds)):
                    conflict = "concordant"
                else:
                    conflict = "discordant"             
            else: 
                if annotation == ">|<":
                    assignment = "downstream"
                    # first consider concordant case
                    if ((s == "+") and (ud < args.flank)) or ((s == "-") and (dd < args.flank)):
                        conflict = "concordant"
                    else:
                        conflict = "discordant"
                elif annotation in [">|>",">|.",".|>"]: 
                    if ud <= dd:
                        assignment = "downstream"
                    else:
                        assignment = "leader"
                    conflict = "concordant" if s == "+" else "discordant"                 
                    # orphan cases are already handled
                elif annotation in ["<|<",".|<","<|."]:
                    if dd <= ud:
                        assignment = "downstream"
                    else:
                        assignment = "leader"
                    conflict = "concordant" if s == "-" else "discordant"
                    # orphan cases are ready handled
                elif annotation == "<|>":
                    # either leader or orphan, orphan cases are already handled
                    assignment = "leader"
                    if ((s == "+") and (dd < args.flank)) or ((s == "-") and (ud < args.flank)):
                        conflict = "concordant"
                    else:
                        conflict = "discordant"                
                else:
                    assignment = "unassigned"
                    conflict = "unknown"
        print(seq_id, start, end,
              names[(seq_id, start, end)], scores[(seq_id, start, end)], s, 
              annotation, distance, conflict, assignment, gene_id,  sep="\t",file=fout)
        if (seq_id, start, end) in antisense_scores:
            print(seq_id, start, end,
                  antisense_names[(seq_id, start, end)], antisense_scores[(seq_id, start, end)], {"+":"-","-":"+"}[s],
                  annotation, distance, {"concordant":"discordant","discordant":"concordant","unknown":"unknown"}[conflict], assignment, gene_id, sep="\t",file=fout)
             
    fout.close()
    logger.info("all done .")

if __name__ == "__main__":
    main() 
        
    

    
     
