#!/usr/bin/env python
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import argparse
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("TNF2cutoff")


def main():
    parser = argparse.ArgumentParser(description='given TNF, predict cutoff to achieve a specified FPR')
    parser.add_argument('--scores', '-s', type=str, required=True, help="score at given FPR")
    parser.add_argument('--tetramer-frequency','-tnf',required=True,help="path of tetramer frequency")
    parser.add_argument('--model','-m',default="model/fpr01.pkl",help="which model to use")
    args = parser.parse_args()
    logger.info("load tetramer frequency ...")
    frequency = pd.read_csv(args.tetramer_frequency,sep="\t",index_col=0)
    logger.info("load model ...")
    model = pickle.load(open(args.model,"rb")) 
    logger.info("run prediction ...")
    y_pred = model["regressor"].predict(frequency.values)
    y_pred = y_pred*model["scaler"] + model["offset"]
    
    genome_ids = list(frequency.index)


    logger.info("saving results ...")
    fout = open(args.scores,"w")
    print("genome id","score",sep="\t",file=fout)
    for genome_id, score in zip(genome_ids, y_pred):
        print(genome_id, round(score,4), sep="\t",file=fout)
    fout.close()

    logger.info("all done .")


if __name__ == "__main__":
    main() 

